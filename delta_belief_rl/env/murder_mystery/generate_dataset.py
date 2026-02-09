"""
adapted from:
https://github.com/tajwarfahim/paprika/blob/main/llm_exploration/game/game_configs/murder_mystery.json
"""

import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import pandas as pd
import yaml

# System prompt template; will be overridden by JSON "agent" template if available
SYSTEM_PROMPT = """You are playing the role of a detective in a murder mystery game. 
The setup for the game is: 
1. You will be provided with a scenario describing a crime and its key elements. Your goal is to solve the mystery by asking questions, examining evidence, and drawing logical conclusions. 
2. For every action you take or question you ask, you will receive feedback from the game. 
3. Your questions and actions should be precise and logical, aimed at uncovering clues, verifying alibis, and piecing together the sequence of events. You should stretegically choose the next action, given the information you have already obtained from the game, and choose actions that lets you catch the culprit as quickly as possible. 
4. You can only take a single action at every turn. 
5. You have to consider all pieces of information, and scrutinize all the characters in the game, including the witnesses or background characters, since the true culprit maybe a witness or a background character, and might not always be one of the primary suspects declared at the beginning of the game. Do not focus on any character too early in the game, rather try to see if anyone's statements are contradictory. 
6. You should always gather enough information before making a decision --- try not to make a mistake! You should also keep your mind open about who can be the true culprit and try to be information seeking, without being too narrowed down on one suspect too quickly. 
7. Once you believe you have enough evidence, you may state your conclusion about the case, which will terminate the game. 
The game starts now. The particular scenario you have is: "{agent}"
Now, take an action to investigate the crime scene or ask a question to the characters involved to solve the mystery:    
"""


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def generate_data_from_json(
    items: list[dict], split: str, N: int | None = None
) -> pd.DataFrame:
    """Build dataframe rows from JSON entries (train/eval).

    Each item is expected to have keys: "agent" (scenario) and "env" (solution).
    """
    dataset = []
    total = len(items) if N == -1 or N is None else min(N, len(items))

    for i in range(total):
        entry = items[i]
        agent_msg = entry.get("agent", "")
        env_msg = entry.get("env", "")
        sample = {
            "data_source": "murder_mystery",
            "id": f"{split}_{i}",
            "golden_answers": env_msg,
            "prompt": [
                {
                    "role": "user",
                    "content": SYSTEM_PROMPT.format(agent=agent_msg),
                }
            ],
            "ability": "reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": [env_msg]},
            },
            "extra_info": {
                "split": split,
                "index": i,
            },
        }
        dataset.append(sample)

    return pd.DataFrame(dataset)


def generate_dataset(config_path: str):
    # Check config file
    if not os.path.exists(config_path):
        sys.exit(f"Error: 'config.yaml' not found in {config_path}.")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config = dict_to_namespace(config_dict)

    # Resolve JSON path and load scenarios
    json_path = os.path.abspath(config.json_path)
    if not os.path.exists(json_path):
        sys.exit(f"Error: customer_service JSON not found at {json_path}")

    with open(json_path, "r") as f:
        json_data = json.load(f)

    split_to_items = {
        "train": json_data.get("train", []),
        "test": json_data.get("eval", []),
    }

    os.makedirs(config.output_path, exist_ok=True)

    for split, split_data in config_dict["data"].items():
        items = split_to_items.get(split)
        if items is None:
            sys.exit(
                f"Error: split '{split}' not found in JSON (expected keys: {list(split_to_items.keys())})"
            )

        df = generate_data_from_json(items, split, split_data["N"])
        print(df.head())

        df.to_parquet(os.path.join(config.output_path, split_data["output_name"]))


def inject_system_prompts(json_path: str, output_path: str | None = None):
    """Load customer_service.json and add SYSTEM_PROMPT to each eval item.

    SYSTEM_PROMPT is the "agent" template at the top of the JSON, formatted with
    the scenario-specific agent string for each eval entry.
    """
    if not os.path.exists(json_path):
        sys.exit(f"Error: JSON file not found at {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    base_prompt = data.get("agent")
    if not base_prompt:
        sys.exit("Error: No 'agent' system prompt found in JSON.")

    eval_items = data.get("eval", [])
    if not isinstance(eval_items, list):
        sys.exit("Error: 'eval' section is missing or not a list.")

    for item in eval_items:
        agent_msg = item.get("agent", "")
        item["SYSTEM_PROMPT"] = base_prompt.format(agent=agent_msg)

    out_path = output_path or json_path
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Wrote updated JSON with SYSTEM_PROMPT to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate dataset for the customer service game or augment JSON with system prompts."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file."
    )
    args = parser.parse_args()

    generate_dataset(args.config)
