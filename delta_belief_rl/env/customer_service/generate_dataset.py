"""
adapted from:
https://github.com/tajwarfahim/paprika/blob/main/llm_exploration/game/game_configs/customer_service.json
"""

import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import pandas as pd
import yaml

# System prompt template; will be overridden by JSON "agent" template if available
SYSTEM_PROMPT = """You are going to role-play as a customer service agent and you have to help a customer resolve their issue. Your goal is to gather enough information to diagnose the problem and provide solution. 
Your instructions are the following: 
1. You will need to ask targeted questions or suggest particular actions to the customer to gather the necessary details. 
2. The customer may not be technically inclined, so keep your language simple and clear. 
3. Avoid making assumptions — ask specific questions to determine the potential causes. You should guide the customer through basic troubleshooting steps and gather data on the situation. 
4. Refine your questions in a strategic way based on the customer's responses for earlier questions. 
5. You should ask questions in an efficient manner, to make the customer satisfied and resolve their problem as quickly as possible. You should also keep your responses short and concise. 
6. If the customer mentions a specific product they are using (for example, ABC electronics), then you are the customer support agent for that product/company, i.e., you represent that product or company and have to take appropriate actions without referring the customer to somewhere else. 
7. Only ask one question at a time to the customer.
Your specific scenario is this: {agent} 
Please start helping the customer now by asking your first question.
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
            "data_source": "customer_service",
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
                "scenario": agent_msg,
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
