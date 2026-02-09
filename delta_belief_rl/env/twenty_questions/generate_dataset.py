"""
for policy training verl required .parquet file format with prompt as data field
details here:
https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html#define-the-data
"""

import os
import yaml
from types import SimpleNamespace
import sys
import pandas as pd
from delta_belief_rl.llm_agent.prompts import (
    DIRECT_PROMPT,
    COT_PROMPT,
    THINKING_PROMPT,
    SYSTEM_PROMPT_ORIGINAL,
    SYSTEM_PROMPT_ADJ,
)
from argparse import ArgumentParser


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def generate_data_dict(
    gt: str,
    split: str,
    cot: bool = True,
    thinking: bool = False,
    system_prompt: bool = True,
    N: int = None,
) -> pd.DataFrame:
    # generate the dict for the dataset
    dataset = []
    N = len(gt) if N == -1 else N

    if system_prompt:
        SYSTEM_PROMPT = SYSTEM_PROMPT_ORIGINAL
    else:
        SYSTEM_PROMPT = SYSTEM_PROMPT_ADJ

    if cot:
        prompt = COT_PROMPT
    elif thinking:
        prompt = THINKING_PROMPT
    else:
        prompt = DIRECT_PROMPT

    for i in range(N):
        # train sample
        sample = {
            "data_source": "tw_q",
            "id": split + "_" + str(i),
            "golden_answers": gt[i],
            "system_prompt": {"role": "system", "content": SYSTEM_PROMPT},
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "reasoning",
            "reward_model": {"style": "rule", "ground_truth": {"target": [gt[i]]}},
            "extra_info": {"split": split, "index": i},
        }
        dataset.append(sample)
    return pd.DataFrame(dataset)


def generate_dataset(config_path: str):
    # Check if the config file exists in the same folder
    if not os.path.exists(config_path):
        sys.exit(f"Error: 'config.yaml' not found in {config_path}.")

    # Load the YAML file
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert the dictionary to an object with attribute access
    config = dict_to_namespace(config_dict)

    assert not (config.cot and config.thinking), (
        "Cannot have both cot and thinking true."
    )

    # Make parquet file data folder
    os.makedirs(config.output_path, exist_ok=True)

    # Load the possible secrets
    for split, split_data in config_dict["data"].items():
        split_path = os.path.join(config.input_path, split_data["input_name"])

        with open(os.path.abspath(split_path), "r") as file:
            gt_list = file.read().splitlines()

        df = generate_data_dict(
            gt_list,
            split,
            config.cot,
            config.thinking,
            config.system_prompt.original,
            split_data["N"],
        )
        print(df.head())

        df.to_parquet(os.path.join(config.output_path, split_data["output_name"]))

        if config.to_jsonl:
            df.to_json(
                os.path.join(
                    config.output_path,
                    split_data["output_name"].replace(".parquet", ".jsonl"),
                ),
                orient="records",
                lines=True,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate dataset for the 20 questions game.")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file."
    )
    args = parser.parse_args()
    generate_dataset(args.config)
