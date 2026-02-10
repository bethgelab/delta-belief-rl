#!/bin/bash
### Example evaluation script for 20 Questions (test set)

# --------- Specify below according to your cluster ---------
#for runpod have to manually set the cuda visible devices
export CUDA_VISIBLE_DEVICES=0,1
# (Optional, for nvidia-container-runtime consistency)
export NVIDIA_VISIBLE_DEVICES=0,1
#set manually a slurm job id if running without slurm
export SLURM_JOB_ID=123456
#---------------------------------------------

set -euo pipefail
mkdir -p .local_logs

#initialise ray 
export VERL_LOGGING_LEVEL='WARN'
source scripts/slurm_setup.sh

# Set default values if not already defined
SEED="${1:-42}"
: "${BASE_MODEL:=iaa01/qwen3-4b-elicit-pos-ckpt56}" && export BASE_MODEL
: "${ORACLE_MODEL:=Qwen/Qwen3-14B}" && export ORACLE_MODEL
: "${EXPERIMENT_NAME:=test_hf_pushed_elicit_pos_w01}" && export EXPERIMENT_NAME

export LOGFILE=${EXPERIMENT_NAME}_$(date +%Y-%m-%d_%H-%M-%S).log

echo "Starting test with LOGFILE: $LOGFILE"

PYTHONUNBUFFERED=1 python3 -m train \
    seed="$SEED" \
    multi_turn.debug=false \
    multi_turn.val.n=8 \
    multi_turn.verify_judge.enabled=true \
    multi_turn.verify_judge.methods="[exact, sentence, question, multiple_questions, length]" \
    multi_turn.verify_judge.false_positive_behavior="yes" \
    data.train_files=delta_belief_rl/env/twenty_questions/direct/coca_plus/train_rl.parquet \
    data.val_files=delta_belief_rl/env/twenty_questions/direct/coca_plus/test.parquet \
    actor_rollout_ref.ngpus=1 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_liger=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.rollout.layered_summon=true \
    actor_rollout_ref.rollout.load_format='safetensors' \
    judge_rollout.model.path=$ORACLE_MODEL \
    judge_rollout.ngpus=1 \
    judge_rollout.cot=true \
    judge_rollout.thinking=false \
    judge_rollout.rollout.response_length=250 \
    judge_rollout.rollout.free_cache_engine=false \
    judge_rollout.rollout.max_num_batched_tokens=2048 \
    judge_rollout.rollout.gpu_memory_utilization=0.7 \
    trainer.validation_data_dir=test/$EXPERIMENT_NAME \
    trainer.rollout_data_dir=rollout/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS_COUNT \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_only=true \
    trainer.val_before_train=true \
    trainer.resume_mode='disable'\
    2>&1 | tee .local_logs/$LOGFILE
