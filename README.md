<div align="center">

# Intrinsic Credit Assignment for Long Horizon Interaction

**[Ilze Amanda Auzina](https://ilzeamandaa.github.io/)\*,
[Joschka Strüber](https://joschkastrueber.github.io/)\*,
[Sergio Hernández-Gutiérrez](https://sergiohg.com/)\*,
[Shashwat Goel](https://shash42.github.io/)†,
[Ameya Prabhu](https://ameya.prabhu.be/)†,
[Matthias Bethge](https://bethgelab.org/)†**

*\* Equal contribution &emsp; † Equal supervision*

**[Tübingen AI Center](https://tuebingen.ai/) · [University of Tübingen](https://uni-tuebingen.de/) · [MPI for Intelligent Systems](https://www.tuebingen.mpg.de/10020/mpi-fuer-intelligente-systeme) · [ELLIS Institute Tübingen](https://institute-tue.ellis.eu/)**

[![alphaXiv](https://img.shields.io/badge/alphaXiv-Paper-%23c83232)](https://www.alphaxiv.org/abs/intrinsic-credit-assignment)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://bethgelab.github.io/delta-belief-rl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

<p align="center">
  <img src="static/images/figure1.png" alt="Main contributions overview" width="60%">
</p>

## TL;DR

How can we train agents to navigate uncertainty over long horizons? We propose **ΔBelief-RL**, which uses the change in an agent's own belief about the correct answer as a dense intrinsic reward for training in long-horizon, information-seeking tasks. No critic or process reward model needed — just the agent's self-assessment. Our trained agents (**CIA** — Curious Information-seeking Agents) outperform DeepSeek-V3.2 (670B) with ~98% fewer parameters on our in-distribution test-set, generalize to out-of-distribution tasks, and continue improving beyond the training horizon.

## Method

ΔBelief-RL leverages an agent's internal beliefs as an intrinsic reward signal, enabling dense credit assignment in probabilistic, long-horizon tasks. In a multi-turn interaction setting where an agent must uncover a latent target concept through a trajectory of actions and observations:

**1. Agent Beliefs** — We elicit the agent's internal belief by leveraging its underlying token probability distribution over the target concept at each turn.

**2. ΔBelief Reward** — The per-turn belief change is computed as:

$$\Delta\text{Belief}_t = \log b_t - \log b_{t-1}$$

**3. Training Reward** — The sparse outcome reward is augmented with the dense intrinsic signal:

$$r_t = \underbrace{r^{\mathrm{eog}}}_{\text{trajectory outcome}} + \underbrace{\lambda\, \max(\Delta\text{Belief}_t, 0)}_{\text{intrinsic exploration}} + \underbrace{r_p}_{\text{efficiency penalty}}$$

We use **Turn-wise GRPO**, computing advantages at the turn level rather than applying the same reward across the entire trajectory.

<p align="center">
  <img src="static/images/beliefs_deepseek_v3.2_base_only.png" alt="Belief updates over time" width="50%">
</p>
<p align="center"><em>Beliefs steadily increase on average, and the rate of growth strongly correlates with the outcome of the trajectory.</em></p>

## Key Results

### CIA outperforms much larger models on 20 Questions

<div align="center">

| Method | Mean@8 ± std | Pass@8 |
|:---|:---:|:---:|
| Baseline (1.7B) | 9.97% ± 1.04% | 32.03% |
| StarPO (1.7B) | 16.54% ± 1.32% | 45.73% |
| **CIA (ours, 1.7B)** | **24.80% ± 1.10%** | **53.10%** |
| Baseline (4B) | 13.34% ± 1.05% | 36.87% |
| StarPO (4B) | 24.36% ± 1.18% | 59.12% |
| **CIA (ours, 4B)** | **33.72% ± 1.26%** | **63.97%** |
| DeepSeek-V3.2 (670B) | 14.35% ± 0.87% | 47.34% |
| Qwen3-235B-A22B-Instr. | 8.83% ± 0.87% | 27.71% |

</div>

### Efficient exploration

<p align="center">
  <img src="static/images/num_questions.png" alt="Number of questions during training" width="50%">
  <br>
  <img src="static/images/num_repeated_questions.png" alt="Repeated questions during training" width="50%">
</p>
<p align="center"><em>ΔBelief-RL reduces the number of turns required and suppresses redundant queries more rapidly than standard GRPO.</em></p>

### Test-time interaction scaling

<p align="center">
  <img src="static/images/1.7b_interaction_scaling_1.png" alt="Test-time interaction scaling 1.7B" width="50%">
  <br>
  <img src="static/images/4b_interaction_scaling_1.png" alt="Test-time interaction scaling 4B" width="50%">
</p>
<p align="center"><em>CIA continues to improve as the interaction budget extends beyond the 20-turn training horizon. At 4B scale, the improvement from 20 to 50 turns is +26% for CIA vs +13% for StarPO.</em></p>

### Out-of-distribution generalization

<p align="center">
  <img src="static/images/passk_benchmarks_k8_games.png" alt="OOD generalization" width="55%">
</p>
<p align="center"><em>Despite training only on 20 Questions, CIA generalizes to Guess My City and Murder Mystery.</em></p>

<p align="center">
  <img src="static/images/passk_cs_k8.png" alt="Customer Service" width="32%">
  <img src="static/images/user_personalization.png" alt="User Personalization" width="30%">
</p>
<p align="center"><em>CIA outperforms StarPO on practical applications: Customer Service (+5–11%) and User Personalization (up to +15%).</em></p>

## Installation

### Prerequisites

- Python >= 3.12
- CUDA 12.8 compatible GPU(s) (minimum 2x GPUs recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone with submodules (includes verl framework)
git clone --recurse-submodules https://github.com/bethgelab/delta-belief-rl.git
cd delta-belief-rl

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
cd delta_belief_rl && uv sync && cd ..
```

### Weights & Biases (optional)

Training logs to W&B by default. Set your credentials:
```bash
export WANDB_PROJECT=delta-belief-rl
export WANDB_ENTITY=<your-entity>
```

## Running Experiments

All training and evaluation scripts are located in `delta_belief_rl/scripts/`. Each script handles Ray initialization and GPU setup automatically via `scripts/slurm_setup.sh`.

### Training

**ΔBelief-RL** (our method):
```bash
# Qwen3-1.7B
sbatch delta_belief_rl/scripts/train/delta-belief-rl/qwen1.7b.sh

# Qwen3-4B
sbatch delta_belief_rl/scripts/train/delta-belief-rl/qwen4b.sh
```

**GRPO baseline** (StarPO):
```bash
# Qwen3-1.7B
sbatch delta_belief_rl/scripts/train/grpo/qwen1.7b.sh

# Qwen3-4B
sbatch delta_belief_rl/scripts/train/grpo/qwen4b.sh
```

### Evaluation

Evaluate trained checkpoints (or HuggingFace-hosted models) on various benchmarks:

```bash
# 20 Questions (in-distribution)
slurm delta_belief_rl/scripts/eval/twenty_questions.sh

# Guess My City (OOD)
slurm delta_belief_rl/scripts/eval/guess_my_city.sh

# Murder Mystery (OOD)
slurm delta_belief_rl/scripts/eval/murder_mystery.sh

# Customer Service (OOD, uses API judge)
slurm delta_belief_rl/scripts/eval/customer_service.sh
```

### GPU Configuration

Adjust `CUDA_VISIBLE_DEVICES` at the top of each script to match your setup. The number of GPUs per model is configured via:
```bash
actor_rollout_ref.ngpus=1 # GPUs for the questioner model
judge_rollout.ngpus=1     # GPUs for the judge model
```

### SLURM Clusters

The setup script (`scripts/slurm_setup.sh`) handles Ray initialization, port allocation, and GPU memory checks automatically. For SLURM-managed clusters, the `SLURM_JOB_ID` is detected automatically; for other environments, set it manually in the script.

## Project Structure

```
delta-belief-rl/
├── train.py                          # Main entry point
├── delta_belief_rl/
│   ├── config/                       # Hydra YAML configs
│   ├── env/                          # Environment data & prompts
│   │   ├── twenty_questions/         #   20 Questions (training env)
│   │   ├── guess_my_city/            #   Guess My City (OOD eval)
│   │   ├── murder_mystery/           #   Murder Mystery (OOD eval)
│   │   └── customer_service/         #   Customer Service (OOD eval)
│   ├── llm_agent/
│   │   ├── generation.py             #   Multi-turn conversation logic
│   │   ├── belief.py                 #   ΔBelief computation
│   │   └── prompts.py                #   Prompt templates per environment
│   ├── trainer/
│   │   ├── multistep_trainer.py      #   Ray-based FSDP training loop
│   │   └── ppo/core_algos.py         #   PPO, GRPO, Turn-GRPO, RLOO
│   ├── workers/                      #   Ray actor & rollout workers
│   ├── scripts/
│   │   ├── train/                    #   Training scripts
│   │   │   ├── delta-belief-rl/      #     ΔBelief-RL (ours)
│   │   │   └── grpo/                 #     GRPO baseline
│   │   └── eval/                     #   Evaluation scripts
│   └── pyproject.toml                # Python dependencies
├── verl/                             # veRL framework (git submodule)
└── static/                           # Project page assets
```

## Citation

```bibtex
@article{cia2026,
  title  = {Intrinsic Credit Assignment for Long Horizon Interaction},
  author = {Auzina, Ilze Amanda and Str\"uber, Joschka and Hern\'andez-Guti\'errez, Sergio and Goel, Shashwat and Prabhu, Ameya and Bethge, Matthias},
  year   = {2026}
}
```

## Acknowledgments

This codebase builds on [veRL](https://github.com/volcengine/verl) for distributed RL training with FSDP and Ray.
