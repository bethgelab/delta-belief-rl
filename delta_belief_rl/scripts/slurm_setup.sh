#!/bin/bash
# Shared setup for all SLURM jobs

set -euo pipefail

# ---- Force NVIDIA stack; avoid ROCm conflicts ----
# VERL/Ray will set CUDA_VISIBLE_DEVICES per-actor; ROCR_* must be unset.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
# Optional: make it very obvious in logs
echo "GPU env after cleanup:"
env | grep -E 'CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES' || true
# --------------------------------------------------

mkdir -p .local_logs
echo "Running on node: $(hostname)"

# ------------- First Check: GPU free mem guard -------------

# GPU free mem guard
MIN_FREE_MEM=75000
LOWEST=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ' | awk '{ for(i=1;i<=NF;i++) if(i==1||$i<min) min=$i; } END{print min}')
echo "Detected minimum free GPU memory on this node: ${LOWEST} MiB"
if [ "${LOWEST}" -lt "${MIN_FREE_MEM}" ]; then
    echo "ERROR: Less than ${MIN_FREE_MEM} MiB free on at least one GPU."
    exit 1
fi
echo "GPU memory check passed. Launching job..."
# ----------------------------------------------

# ---------- Second Define a temp dir for Ray ----------
JOBID="${SLURM_JOB_ID:-$$}"
USER_TOP="${USER:-u$(id -u)}"

# Prefer site-provided temp first, then per-user dirs on fast local disks.
CANDIDATES=(
    "/tmp/r/${JOBID}"
    "/dev/shm/r/${JOBID}"
    "${SLURM_TMPDIR:-}"
    "${TMPDIR:-}"
    "/scratch/${USER_TOP}/r/${JOBID}"
    "/scratch/r/${JOBID}"
)

pick_ray_tmp() {
    local needs_suffix="/ray/session_YYYY-mm-dd_HH-MM-SS_XXXXXXXXXXXX/sockets/plasma_store"
    local max=107
    for cand in "${CANDIDATES[@]}"; do
        [ -z "${cand}" ] && continue
        # Only accept if base + suffix fits the kernel sock path limit
        if [ $((${#cand} + ${#needs_suffix})) -lt $max ]; then
            if mkdir -p -m 700 "$cand" 2>/dev/null && touch "$cand/.probe" 2>/dev/null; then
                rm -f "$cand/.probe"
                echo "$cand"
                return 0
            fi
        fi
    done
    return 1
}

RAY_REAL="$(pick_ray_tmp)" || {
    echo "ERROR: No writable temp dir for Ray."
    exit 1
}

# Nice location in your project (just a symlink; safe for long paths)
RAY_HOME="$PWD/.ray_jobs/job_${JOBID}"
mkdir -p "$(dirname "$RAY_HOME")"
ln -sfn "$RAY_REAL" "$RAY_HOME"

# Tell Ray to use the SHORT path (avoids 107-char error)
export RAY_TMPDIR="$RAY_REAL"
export TMPDIR="$RAY_REAL"

# Debug info
echo "Ray temp dir  : $RAY_REAL"
echo "Ray project ln: $RAY_HOME -> $(readlink -f "$RAY_HOME" 2>/dev/null || echo '?')"
echo "Projected socket path length: $(echo -n "$RAY_REAL/session_x/sockets/plasma_store" | wc -c)"
# -------------------------------------------------------------------

#activate virtual environment
source delta_belief_rl/.venv/bin/activate

# ------------- Third Specify env params -------------
# Avoid unbound variable error if LD_LIBRARY_PATH is unset
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Defaults (set before starting ray)
SEED="${1:-42}"
: "${CUDA_VISIBLE_DEVICES:=0,1}" && export CUDA_VISIBLE_DEVICES

# set the W&B env vars before starting ray
: "${WANDB_PROJECT:=delta-belief-rl}" && export WANDB_PROJECT

# Start Ray head (single node) — NOTE: pass --temp-dir
# count only the GPUs Slurm made visible to you
NUM_GPUS_STR="${CUDA_VISIBLE_DEVICES:-}"
NUM_GPUS_COUNT=$([ -z "$NUM_GPUS_STR" ] && echo 0 || echo "$NUM_GPUS_STR" | awk -F, '{print NF}')
echo "Visible GPUs (count): $NUM_GPUS_COUNT"
if [ "$NUM_GPUS_COUNT" -lt 1 ]; then
    echo "ERROR: No GPUs visible via CUDA_VISIBLE_DEVICES; refusing to start."
    exit 1
fi

# to prevemt oom set (for loss.backward() in actor)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# ----------------------------------------------

# ---- Fourth: reserve a clean port block & ensure no collisions ----
# set custom ip address to avoid conflict of 2 jobs on same node
ATTEMPTS=100
HEAD_IP="$(hostname -I | awk '{print $1}')"

# pick all required TCP ports in a safe range, under a node-wide lock. Keep the
# lock file outside the job-specific temp dir so multiple jobs on the same node
# coordinate instead of racing and selecting the same ports. Force a per-user
# location to avoid permission issues on shared /tmp.
USER_LOCK_DIR="/tmp/${USER_TOP}/ray_ports"
mkdir -p "$USER_LOCK_DIR"
PORT_LOCK_FILE="${USER_LOCK_DIR}/ray_ports.lock"

port_free() {
    local port="$1"
    local host="${HEAD_IP:-127.0.0.1}"
    # Return 0 if free, 1 if in use
    if command -v ss >/dev/null 2>&1; then
        if ss -Htanl | awk '$1=="LISTEN"{print $4}' | grep -qE ":${port}$"; then
            return 1
        fi
    elif command -v netstat >/dev/null 2>&1; then
        if netstat -tln 2>/dev/null | awk 'NR>2{print $4}' | grep -qE ":${port}$"; then
            return 1
        fi
    fi
    # Extra safety: attempt to bind via Python (if available)
    if command -v python3 >/dev/null 2>&1; then
        if python3 - "$host" "$port" <<'PY' >/dev/null 2>&1; then
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind((host, port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
            return 0
        else
            return 1
        fi
    fi
    # Last resort: quick connect probe via /dev/tcp (0.2s timeout)
    if command -v timeout >/dev/null 2>&1 && timeout 0.2 bash -c "exec 3<>/dev/tcp/${host}/${port}" 2>/dev/null; then
        exec 3>&- 3<&-
        return 1
    fi
    return 0
}

pick_port_block() {
    local start=7000
    local end=9500
    local tries=200
    for _ in $(seq 1 "$tries"); do
        local base=$((start + (RANDOM % (end - start - 6))))
        local p_gcs=$base
        local p_dash=$((base + 1))
        local p_client=$((base + 2))
        local p_objmgr=$((base + 3))
        local p_node=$((base + 4))
        local p_agent=$((base + 5))
        local p_metrics=$((base + 6))
        local worker_min=$((base + 200))
        local worker_max=$((worker_min + 63))

        if [ "$worker_max" -ge 65500 ]; then
            continue
        fi

        local busy=false
        for port in "$p_gcs" "$p_dash" "$p_client" "$p_objmgr" "$p_node" "$p_agent" "$p_metrics"; do
            if ! port_free "$port"; then
                busy=true
                break
            fi
        done
        if [ "$busy" = false ]; then
            echo "$p_gcs $p_dash $p_client $p_objmgr $p_node $p_agent $p_metrics $worker_min $worker_max"
            return 0
        fi
    done
    return 1
}

for i in $(seq 1 "$ATTEMPTS"); do
    # Acquire node-wide lock while choosing ports to avoid races between jobs using this script
    exec 9>"$PORT_LOCK_FILE"
    if ! flock -w 10 9; then
        echo "[warn] could not acquire port lock; retrying..."
        sleep 1
        continue
    fi

    PORTS=$(pick_port_block) || PORTS=""

    if [ -z "$PORTS" ]; then
        echo "[warn] failed to find free port bundle; retrying..."
        # release lock before retrying
        flock -u 9
        exec 9>&-
        sleep 1
        continue
    fi

    GCS_PORT="$(echo "$PORTS" | awk '{print $1}')"
    DASH_PORT="$(echo "$PORTS" | awk '{print $2}')"
    CLIENT_PORT="$(echo "$PORTS" | awk '{print $3}')"
    OBJECT_MANAGER_PORT="$(echo "$PORTS" | awk '{print $4}')"
    NODE_PORT="$(echo "$PORTS" | awk '{print $5}')"
    AGENT_PORT="$(echo "$PORTS" | awk '{print $6}')"
    METRICS_PORT="$(echo "$PORTS" | awk '{print $7}')"
    WORKER_PORT_MIN="$(echo "$PORTS" | awk '{print $8}')"
    WORKER_PORT_MAX="$(echo "$PORTS" | awk '{print $9}')"
    echo "[ports] trying GCS=$GCS_PORT DASH=$DASH_PORT CLIENT=$CLIENT_PORT OBJMGR=$OBJECT_MANAGER_PORT NODE=$NODE_PORT WORKER=$WORKER_PORT_MIN-$WORKER_PORT_MAX (attempt $i/$ATTEMPTS)"

    if ray start --head \
        --include-dashboard=false \
        --node-ip-address="$HEAD_IP" \
        --port="$GCS_PORT" \
        --dashboard-port="$DASH_PORT" \
        --ray-client-server-port="$CLIENT_PORT" \
        --object-manager-port="$OBJECT_MANAGER_PORT" \
        --node-manager-port="$NODE_PORT" \
        --dashboard-agent-listen-port="$AGENT_PORT" \
        --metrics-export-port="$METRICS_PORT" \
        --min-worker-port="$WORKER_PORT_MIN" \
        --max-worker-port="$WORKER_PORT_MAX" \
        --num-cpus="${SLURM_CPUS_ON_NODE:-16}" \
        --num-gpus="${NUM_GPUS_COUNT:?}" \
        --object-store-memory=$((8 * 1024 ** 3)) \
        --temp-dir="${RAY_TMPDIR:?}"; then
        export RAY_GCS_ADDRESS="$HEAD_IP:$GCS_PORT"
        export RAY_CLIENT_ADDRESS="ray://$HEAD_IP:$CLIENT_PORT"
        echo "Ray start succeeded on attempt $i."
        # release lock after success
        flock -u 9
        exec 9>&-
        break
    else
        echo "[warn] ray start failed; retrying with a new port block..."
        if [ "$i" -eq "$ATTEMPTS" ]; then
            echo "[FATAL] Ray failed to start after $ATTEMPTS attempts."
            # release lock before exiting
            flock -u 9
            exec 9>&-
            exit 1
        fi
        # release lock before retrying
        flock -u 9
        exec 9>&-
        sleep 1
    fi
done

#--------------------------------------------

# Wait for GCS to come up
for i in {1..20}; do
    if ray status --address="$RAY_GCS_ADDRESS" >/dev/null 2>&1; then
        echo "Ray is up."
        break
    fi
    echo "Waiting for Ray... ($i)"
    sleep 2
done
ray status --address="$RAY_GCS_ADDRESS" || {
    echo "Ray failed to start"
    exit 1
}
# ----------------------------------------------
