import modal

app = modal.App("delta-belief-rl")
github_token = modal.Secret.from_name("GITHUB_TOKEN")
wandb_api_key = modal.Secret.from_name("WANDB_API_KEY")
hf_home = modal.Volume.from_name("hf_home", create_if_missing=True)


def get_modal_image():
    modal_image = (
        modal.Image.from_registry(
            "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
        )
        .apt_install("git", "curl", "python3-dev", "build-essential", "bash")
        .pip_install("uv")
        .run_commands(
            "git clone https://token:$GITHUB_TOKEN@github.com/bethgelab/delta-belief-rl.git /root/delta-belief-rl",
            "git -C /root/delta-belief-rl submodule set-url verl https://github.com/volcengine/verl.git",
            "git -C /root/delta-belief-rl submodule update --init --recursive",
            "cd /root/delta-belief-rl/delta_belief_rl && uv venv --python=3.11",
            "cd /root/delta-belief-rl/delta_belief_rl && uv sync",
            "apt-get update && apt-get install -y python3-tk",
            "cd /root/delta-belief-rl/delta_belief_rl && uv pip install pip",
            "cd /root/delta-belief-rl/delta_belief_rl && uv pip install flash-attn==2.7.4.post1 --no-build-isolation",
            "cd /root/delta-belief-rl/delta_belief_rl && uv pip install spacy",
            "cd /root/delta-belief-rl/delta_belief_rl && .venv/bin/python -m spacy download en_core_web_sm",
            "cd /root/delta-belief-rl/verl/ && /root/delta-belief-rl/delta_belief_rl/.venv/bin/python -m pip install -e .",
            "cd /root/delta-belief-rl && /root/delta-belief-rl/delta_belief_rl/.venv/bin/python -m pip install -e .",
            secrets=[github_token],
        )
        .pip_install("wandb")
        .run_commands(
            "wandb login $WANDB_API_KEY --relogin",
            secrets=[wandb_api_key],
        )
        .env(
            {
                "PATH": "/root/delta-belief-rl/delta_belief_rl/.venv/bin:/usr/local/cuda/bin:$PATH",
                "CUDA_HOME": "/usr/local/cuda",
                "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
                "HF_HOME": "/root/.cache/huggingface",
                "WANDB_PROJECT": "delta-belief-rl",
            }
        )
    )
    return modal_image


image = get_modal_image()
