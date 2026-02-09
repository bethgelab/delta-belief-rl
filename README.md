
## To Install the Repository 
clone the repo including its submodules via
```
git clone --recurse-submodules <repository-url>
```

follow instruction in delta_belief_rl/README.md


## General Run Settings
> for an example see _scripts/dense_rewards/qwen3-1.7b-sft.sh_

Always specify:
- the number of GPUs used 
- which questioner model to use 
- which judge model to use 
```
CUDA_VISIBLE_DEVICES:=0,1
BASE_MODEL:=Klingspor/lta_singleturn_hard_sft_qwen3-1.7b
ORACLE_MODEL:=Qwen/Qwen2.5-14B-Instruct
```
Furthmore, you need to manually specify the number of gpus used per model
```
actor_rollout_ref.ngpus=1
judge_rollout.ngpus=1
```
Lastly, specify the experiment name:
```
EXPERIMENT_NAME:=<your_experiment_name>
```
Upon running there will be:
- a log file generated in logs/
- model checkpoints generated in checkpoints/
- model outputs (game rollouts) stored in
    - train_rollout
    - validation


## Spcific Training and Validation Run Details
### 1. Dense Loss with PPO GAE
> for an example script see _scripts/dense_rewards/qwen3-1.7b-sft.sh_

to enable set (1) logprob computation to true, (2) advantage estiamtion as gae, and (3) NOT offload actor model params as we need them iteratively to compute the logprobs also during game play (should be faster than constantly offloading and onloading):
```
multi_turn.logprob_reward.enable=true
algorithm.adv_estimator=gae
actor_rollout_ref.actor.fsdp_config.param_offload=false
```
To specify a different difference computation set (deafault is 'step'):
```
- basewise difference:  multi_turn.logprob_reward.diff_method='base' 
```


last note, currently we are using sum wise aggregation for secrets that span multiple tokens, this can be changed to mean by passing:
```
multi_turn.logprob_reward.agg='mean' \
```

checklist tested runs:
- [x] gae ppo loss with stepwise loss 
- [x] gae ppo loss with basewise loss 


### 2. Sparse Loss with GRPO 
> for an example script see _scripts/sparse_rewards/qwen3-1.7b-sft.sh_

For sparse loss computation with GRPO training (1) disable logprob computation; (2) set number of rollouts for training; (3) set advantage estimation to grpo
```
multi_turn.logprob_reward.enable=false
multi_turn.train.n=5
algorithm.adv_estimator=grpo
```
the default set-up is standard GRPO with normalisation by std, if you want to run Dr. GRPO set:
```
algorithm.norm_adv_by_std_in_grpo=false
```

#### 2.1 DAPO GRPO Training
> for an example script see _scripts/sparse_rewards/qwen3-1.7b-sft-dapo.sh_

if you want to ran DAPO, set: 
```
algorithm.filter_groups.enable=true
```
default parameter for how many time to recompute the batch for a valid sample size is set to 10, can be adjusted via:
```
algorithm.filter_groups.max_num_gen_batches=..
```


#### 2.2. Multi-Step Prompt for GRPO Training
> for an example script see _scripts/sparse_rewards/qwen3-1.7b-sft-mp.sh_

if you want to ran multi-step prompt GRPO training, specify the correct training file as: 
```
data.train_files=delta_belief_rl/env/twenty_questions/direct/multi_step/multiturn_grpo_all_skip_2_easy_unique.parquet
```
and reduce the number of training steps:
```
multi_turn.max_turns.train=5
```

checklist tested runs:
- [x] standard grpo (looks like dapo was running, yes, set the wron param)
- [x] dapo
- [x] multi-step prompt


### 3. Belief computation on pregenerated data 
> for an example script see _scripts/beliefs_only/qwen3-1.7b-sft.sh_
can be ran on a single gpu as only transformer model needed. To only 'activate' the questioner model set (default is 'actor_rollout'):
```
actor_rollout_ref.role='actor'
trainer.belief_only=true 
```
to disable the judge model set:
```
judge_rollout.enable=false
```
spcify which model's generations to use as:
```
data.val_files=delta_belief_rl/env/twenty_questions/direct/multi_step/multiturn_grpo_best_no_skip_test_gemini_unique.parquet 
```

### 4. Multiple GPUs for questioner model:
for runs on multiple gpus, for example 32B model, you also have to adjust that when the 'actor_rollout_ref.actor.ppo_mini_batch_size' is divided by the number of GPUs for the actor is equal to 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu'. 
For an example see _scripts/beliefs_only/qwen3-32b-sft.sh_

checklist tested runs:
- [x] beliefs on gemini generations 

### Other Settings (might be useful)

To get a more detailed ouput for debugging set (default is false):
```
export VERL_LOGGING_LEVEL='DEBUG'
multi_turn.debug=true
```
the training files are set to as:
```
data.train_files=delta_belief_rl/env/twenty_questions/direct/coca/rl_training.parquet
data.val_files=delta_belief_rl/env/twenty_questions/direct/coca/test.parquet
```
to only run validation add:
```
+trainer.val_only=true 
```

default memory utalisation for actor vllm:
```
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 
```

before every train run we run validation first, if you want to disbale this set:
```
trainer.val_before_train=false \
```

for debugging can reduce the total number of training steps:
```
trainer.total_training_steps=2
```

default save and test frequency is 25 
```
trainer.save_freq=25 \
trainer.test_freq=25 \
```

if using qwen3 as judge probably have to increase the allowed response lenght of the judge via:
```
judge_rollout.rollout.response_length=4000 \
```

currently we are using `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95` based on r1-search implementation, haven't tested different values, maybe worth checking


### Updates since 13.06.25.
- [x] Implemented regex reward, default set to false, to activate pass:
```
multi_turn.reward.regex_reward=true
```
- [x] Adjusted judge prompt to include regex check in `learining_to_ask/llm_agent/prompts.py` and in `def _ask_question()` in `generation.py`. Default set to true, can disable via:
```
multi_turn.answer_regex=false
```
- [x] Implemented RLOO for token-level dense rewards check in `learining_to_ask/trainer/ppo/core_algos.py`

    - to use RLOO pass `algorithm.adv_estimator=rloo` see an example script file under `delta_belief_rl/scripts/dense_rewards/qwen3-1.7b-sft-rloo.sh`

- [x] Added an example slurm script that runs one of the scripts specified as `run_qwen3-1.7B-sft.slurm`


