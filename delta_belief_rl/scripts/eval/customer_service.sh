#!/bin/bash
### Example evaluation script for customer service 

# --------- Specify below according to your cluster ---------
#for runpod have to manually set the cuda visible devices
export CUDA_VISIBLE_DEVICES=0
# (Optional, for nvidia-container-runtime consistency)
export NVIDIA_VISIBLE_DEVICES=0
#set manually a slurm job id if running without slurm
export SLURM_JOB_ID=123456
#---------------------------------------------

set -euo pipefail
mkdir -p .local_logs

#initialise ray 
export VERL_LOGGING_LEVEL='WARN'
source delta_belief_rl/scripts/slurm_setup.sh
#IMPORTANT
#remember to set your OpenRouter API key in: delta_belief_rl/workers/rollout/api_key.py


SEED="${1:-42}"
: "${BASE_MODEL:=iaa01/CIA-4B}" && export BASE_MODEL
: "${ORACLE_MODEL:=qwen/qwen3-235b-a22b-2507}" && export ORACLE_MODEL
: "${EXPERIMENT_NAME:=eval_cia_cs}" && export EXPERIMENT_NAME

export LOGFILE=${EXPERIMENT_NAME}_$(date +%Y-%m-%d_%H-%M-%S).log

echo "Starting test with LOGFILE: $LOGFILE"

PYTHONUNBUFFERED=1 python3 -m train \
    seed="$SEED" \
    multi_turn.env="customer_service" \
    multi_turn.debug=true \
    multi_turn.val.n=8 \
    multi_turn.verify_judge.enabled=true \
    multi_turn.verify_judge.methods="[multiple_questions]" \
    data.val_files=delta_belief_rl/env/customer_service/direct/paprika/test.parquet \
    data.system_key=null \
    data.max_prompt_length=3000 \
    data.max_response_length=100 \
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
    +judge_rollout.model.api_model_name=$ORACLE_MODEL \
    +judge_rollout.model.api_batch_size=200 \
    judge_rollout.ngpus=0 \
    judge_rollout.repeated_prompt=false \
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
