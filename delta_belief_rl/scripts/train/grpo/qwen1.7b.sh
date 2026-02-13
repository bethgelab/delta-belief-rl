#!/bin/bash
### Example training script for 20 Questions with Qwen3-1.7B SFT as base model 

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

SEED="${1:-42}"
: "${BASE_MODEL:=Klingspor/Qwen3-1.7B-SFT}" && export BASE_MODEL
: "${ORACLE_MODEL:=Qwen/Qwen3-14B}" && export ORACLE_MODEL
: "${EXPERIMENT_NAME:=train_grpo_qwen1.7b}" && export EXPERIMENT_NAME

export LOGFILE="${EXPERIMENT_NAME}_$(date +%Y-%m-%d_%H-%M-%S).log"

set -x
echo "[INFO] Starting training at $(date) (LOGFILE=$LOGFILE)" | tee -a ".local_logs/$LOGFILE"

PYTHONUNBUFFERED=1 python3 -m train \
    seed="$SEED" \
    multi_turn.logprob_reward.enabled=false \
    multi_turn.logprob_sampling.enabled=false \
    multi_turn.verify_judge.enabled=true \
    multi_turn.verify_judge.methods="[exact, sentence, question, multiple_questions, length]" \
    multi_turn.verify_judge.false_positive_behavior="yes" \
    multi_turn.train.n=16 \
    multi_turn.val.n=4 \
    multi_turn.debug=false \
    algorithm.adv_estimator=grpo \
    data.train_files=learning_to_ask/env/twenty_questions/direct/coca_plus/train_rl.parquet \
    data.val_files=learning_to_ask/env/twenty_questions/direct/coca_plus/val.parquet \
    data.train_batch_size=240 \
    actor_rollout_ref.ngpus=1 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.adapters="[{name: actor_lora, r: 64, alpha: 64, target_modules: all-linear}]" \
    actor_rollout_ref.model.use_liger=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=80 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.loss_agg_mode='seq-mean-token-mean' \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    +actor_rollout_ref.actor.optim.betas="[0.9,0.95]" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=70000 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.rollout.layered_summon=true \
    actor_rollout_ref.rollout.load_format='safetensors' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    judge_rollout.model.path=$ORACLE_MODEL \
    judge_rollout.ngpus=1 \
    judge_rollout.cot=true \
    judge_rollout.thinking=false \
    judge_rollout.rollout.response_length=250 \
    judge_rollout.rollout.free_cache_engine=false \
    judge_rollout.rollout.max_num_batched_tokens=8192 \
    judge_rollout.rollout.gpu_memory_utilization=0.7 \
    trainer.total_epochs=15 \
    trainer.validation_data_dir=validation/$EXPERIMENT_NAME \
    trainer.val_before_train=true \
    trainer.rollout_data_dir=rollout/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS_COUNT \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.save_freq=4 \
    trainer.test_freq=4 \
    trainer.resume_mode="auto" \
    2>&1 | tee -a .local_logs/$LOGFILE