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
from argparse import ArgumentParser

# System prompt from guess_my_city environment under papria configuration

SYSTEM_PROMPT = """You are playing a game called 'Guess My City.' Your task is to guess the home city of the other player within 20 questions.

You must follow these rules:
1. You may ask open-ended questions about the city's characteristics, such as geography, culture, landmarks, cuisine, climate, or other unique traits.
2. Do not ask directly for the name of the city or country.
3. Use the answers from the other player to strategically decide the next question, your goal is to arrive at the correct city with as few questions as possible. 
4. After gathering enough information, you can attempt to guess the city, but each guess counts as one of your 20 questions.

Example questions: 'What is your favorite cuisine from your home city?', 'Is your home city located near the coastline?', 'What kind of landmarks is your city known for?' 
Your goal is to identify the city through deduction, pattern recognition, and strategic questioning.

5. When you are confident, make a guess, by mentioning the name of the city and the country it is in, for example, 'Is the city Jakarta, Indonesia?'

The game starts now, and start asking your questions.
"""


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def generate_data_dict(
    gt: str, split: str, cot: bool = True, system_prompt: bool = True, N: int = None
) -> pd.DataFrame:
    # generate the dict for the dataset
    dataset = []
    N = len(gt) if N == -1 else N

    if system_prompt:
        orig_prompt = SYSTEM_PROMPT
    else:
        raise NotImplementedError(
            "Only original system prompt is implemented for guess_my_city."
        )

    for i in range(N):
        # train sample
        sample = {
            "data_source": "guess_my_city",
            "id": split + "_" + str(i),
            "golden_answers": gt[i],
            "prompt": [{"role": "user", "content": orig_prompt}],
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

    # Make parquet file data folder
    os.makedirs(config.output_path, exist_ok=True)

    # Load the possible secrets
    for split, split_data in config_dict["data"].items():
        split_path = os.path.join(config.input_path, split_data["input_name"])

        with open(os.path.abspath(split_path), "r") as file:
            gt_list = file.read().splitlines()

        df = generate_data_dict(
            gt_list, split, config.cot, config.system_prompt.original, split_data["N"]
        )
        print(df.head())

        df.to_parquet(os.path.join(config.output_path, split_data["output_name"]))


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate dataset for the guess my city game.")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file."
    )
    args = parser.parse_args()
    generate_dataset(args.config)
