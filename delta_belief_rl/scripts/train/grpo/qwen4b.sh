#!/bin/bash
### Example training script for 20 Questions with Qwen3-4B SFT as base model 

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
: "${BASE_MODEL:=Klingspor/lta_singleturn_hard_sft_qwen3-4b}" && export BASE_MODEL
: "${ORACLE_MODEL:=Qwen/Qwen3-14B}" && export ORACLE_MODEL
: "${EXPERIMENT_NAME:=train_grpo_qwen4b}" && export EXPERIMENT_NAME

export LOGFILE="${EXPERIMENT_NAME}_$(date +%Y-%m-%d_%H-%M-%S).log"

set -x
echo "[INFO] Starting training at $(date) (LOGFILE=$LOGFILE)" | tee -a ".local_logs/$LOGFILE"

