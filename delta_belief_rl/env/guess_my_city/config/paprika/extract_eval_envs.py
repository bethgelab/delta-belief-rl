#!/usr/bin/env python3
"""Extract all 'env' values from the 'eval' section of guess_my_city.json and write to eval.txt."""

import json
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    json_path = script_dir / "guess_my_city.json"
    output_path = script_dir / "eval.txt"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    eval_envs = [item["env"] for item in data.get("eval", []) if "env" in item]

    with open(output_path, "w", encoding="utf-8") as f:
        for env in eval_envs:
            f.write(env + "\n")

    print(f"Extracted {len(eval_envs)} env values to {output_path}")

if __name__ == "__main__":
    main()
