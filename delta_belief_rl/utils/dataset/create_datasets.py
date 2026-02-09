import os
import json
import random
import pandas as pd


random.seed(42)


def split_secrets_and_data(
    input_paths: list[str],
    output_secrets_dir: str,
    output_data_dir: str,
    frac_rl_train: float = 0.75,
    frac_sft_train: float = 0.1,
    frac_val: float = 0.05,
    frac_test: float = 0.1,
):
    """
    Reads multiple JSONL files, checks which secrets were solved at least once and splits
    the successful secrets into training for RL, SFT, validation, and test sets.

    Args:
        input_paths (list[str]): List of paths to the input JSONL files.
        output_secrets_dir (str): Path to save the secrets JSONL file.
        output_data_dir (str): Path to save the data JSONL file.
        frac_rl_train (float): Fraction of data for RL training.
        frac_sft_train (float): Fraction of data for SFT training.
        frac_val (float): Fraction of data for validation.
        frac_test (float): Fraction of data for testing.
    """
    # Create the output directories if they do not exist
    os.makedirs(output_secrets_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)

    # Set the output paths for secrets and data
    solved_secrets_path = os.path.join(output_secrets_dir, "secrets.txt")
    unsolved_secrets_path = os.path.join(output_secrets_dir, "unsolved_secrets.txt")
    train_sft_secrets_path = os.path.join(output_secrets_dir, "train_sft_secrets.txt")
    train_rl_secrets_path = os.path.join(output_secrets_dir, "train_rl_secrets.txt")
    val_secrets_path = os.path.join(output_secrets_dir, "val_secrets.txt")
    test_secrets_path = os.path.join(output_secrets_dir, "test_secrets.txt")
    train_sft_data_path = os.path.join(output_data_dir, "train_sft_data.parquet")
    train_sft_data_best_path = os.path.join(
        output_data_dir, "train_sft_data_best.parquet"
    )
    train_rl_data_path = os.path.join(output_data_dir, "train_rl_data.parquet")
    val_data_path = os.path.join(output_data_dir, "val_data.parquet")
    test_data_path = os.path.join(output_data_dir, "test_data.parquet")

    # Load all input data
    all_data = load_inference_data(input_paths)

    # For each secret, check if it was solved at least once
    all_secrets = set()
    secrets = set()
    for item in all_data:
        all_secrets.add(item["secret"])
        if item.get("game_status") is True:
            secrets.add(item["secret"])

    # Split the secrets into those that were solved and those that were not
    unsolved_secrets = list(all_secrets - secrets)
    unsolved_secrets.sort()
    secrets = list(secrets)
    train_rl_secrets, train_sft_secrets, val_secrets, test_secrets = split_secrets(
        secrets, frac_rl_train, frac_sft_train, frac_val, frac_test
    )

    # Filter the data for each split
    train_rl_data = [item for item in all_data if item["secret"] in train_rl_secrets]
    train_sft_data = [item for item in all_data if item["secret"] in train_sft_secrets]
    val_data = [item for item in all_data if item["secret"] in val_secrets]
    test_data = [item for item in all_data if item["secret"] in test_secrets]

    # Select the SFT data with the lowest or second lowest number of turns
    train_sft_data_best = []
    for secret in train_sft_secrets:
        secret_data = [
            item
            for item in train_sft_data
            if (item["secret"] == secret and item["game_status"] is True)
        ]
        if secret_data:
            # Sort by number of turns (length of history)
            sorted_data = sorted(secret_data, key=lambda x: len(x["history"]))
            # Select the one with the lowest number of turns
            train_sft_data_best.extend(
                sorted_data[:2]
            )  # Take the best two if available

    print(f"Selected {len(train_sft_data_best)} best SFT data entries.")

    # Save the secrets to their respective files
    save_secrets(secrets, solved_secrets_path)
    save_secrets(unsolved_secrets, unsolved_secrets_path)
    save_secrets(train_rl_secrets, train_rl_secrets_path)
    save_secrets(train_sft_secrets, train_sft_secrets_path)
    save_secrets(val_secrets, val_secrets_path)
    save_secrets(test_secrets, test_secrets_path)

    # Save the data to their respective files
    save_data(train_rl_data, train_rl_data_path)
    save_data(train_sft_data, train_sft_data_path)
    save_data(train_sft_data_best, train_sft_data_best_path)
    save_data(val_data, val_data_path)
    save_data(test_data, test_data_path)

    # Print the number of secrets in each split
    print(f"Total solved secrets: {len(secrets)}")
    print(f"Unsolved secrets: {len(unsolved_secrets)}")
    print(f"Train RL secrets: {len(train_rl_secrets)}")
    print(f"Train SFT secrets: {len(train_sft_secrets)}")
    print(f"Validation secrets: {len(val_secrets)}")
    print(f"Test secrets: {len(test_secrets)}")


def split_secrets(
    secrets: list[str],
    frac_rl_train: float = 0.6,
    frac_sft_train: float = 0.1,
    frac_val: float = 0.1,
    frac_test: float = 0.2,
) -> tuple[list[str], list[str], list[str], list[str]]:
    # Shuffle the solved secrets
    random.shuffle(secrets)

    # Calculate the number of secrets for each split
    total_solved = len(secrets)
    num_rl_train = int(total_solved * frac_rl_train)
    num_sft_train = int(total_solved * frac_sft_train)
    num_val = int(total_solved * frac_val)

    # Split the solved secrets
    train_rl_secrets = secrets[:num_rl_train]
    train_sft_secrets = secrets[num_rl_train : num_rl_train + num_sft_train]
    val_secrets = secrets[
        num_rl_train + num_sft_train : num_rl_train + num_sft_train + num_val
    ]
    test_secrets = secrets[num_rl_train + num_sft_train + num_val :]

    # Sort the splits
    train_rl_secrets.sort()
    train_sft_secrets.sort()
    val_secrets.sort()
    test_secrets.sort()

    return (train_rl_secrets, train_sft_secrets, val_secrets, test_secrets)


def convert_to_singleturn_sft_format(
    input_path: str,
    output_path: str,
    history_key: str = "history",
    prompt_key: str = "prompt",
    system_key: str = "system_prompt",
    response_key: str = "response",
    skip_last_turns: int = 0,
    split: str = "train",
    filter: bool = False,
) -> None:
    """
    Convert parquet files containing conversation data into a parquet file
    compatible with verl's ChatSingleTurnSFTDataset.

    Args:
        input_path: Path to a parquet files
        output_path: Path to save the output parquet file
        history_key: Key for the conversation (default: "history")
        prompt_key: Key for the user prompt (default: "prompt")
        system_key: Key for the system prompt (default: "system_prompt")
        response_key: Key for the assistant response (default: "response")
        skip_last_turns: Number of last turns to skip in the conversation (default: 0)
        split: Split name for the dataset (default: "train")
        filter: Whether to filter the dataset to one random turn per game (default: False)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read all data from input file
    if input_path.endswith(".jsonl"):
        all_data = load_jsonl(input_path)
    elif input_path.endswith(".parquet"):
        all_data = pd.read_parquet(input_path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    # Save copies of each game after each question
    all_conversations = []
    for i, conversation in enumerate(all_data):
        # Get the messages from the conversation
        messages = list(conversation[history_key])

        # Set the role to "user" and "assistant"
        for j in range(1, len(messages), 2):
            messages[j]["role"] = "user"
            if j + 1 < len(messages):
                messages[j + 1]["role"] = "assistant"

        # Create a new conversation for each question
        num_questions = (
            len(messages) - 1
        ) // 2  # Each question has a response, ignore system prompt
        if filter:
            # Select one random question to create a single-turn dataset
            if num_questions - skip_last_turns <= 0:
                continue
            i = random.randint(0, num_questions - skip_last_turns - 1)
            new_conversation = {
                "data_source": "tw_q",
                "id": split + "_" + str(i),
                "golden_answers": conversation["secret"],
                "source": conversation.get("source", "unknown"),
                "finished_at": conversation["finished_at"],
                "game_status": conversation.get("game_status", False),
                prompt_key: messages[1 : 2 * i + 2],
                response_key: messages[2 * i + 2]["content"],
                system_key: messages[0] if system_key else {},
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": conversation["secret"]},
                },
                "extra_info": {"split": split, "index": i},
            }

            all_conversations.append(new_conversation)

            continue

        for i in range(num_questions - skip_last_turns):
            # Create a new conversation with the first i questions and responses
            new_conversation = {
                "data_source": "tw_q",
                "id": split + "_" + str(i),
                "golden_answers": conversation["secret"],
                "source": conversation.get("source", "unknown"),
                "finished_at": conversation["finished_at"],
                "game_status": conversation.get("game_status", False),
                prompt_key: messages[1 : 2 * i + 2],
                response_key: messages[2 * i + 2]["content"],
                system_key: messages[0] if system_key else {},
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": conversation["secret"]},
                },
                "extra_info": {"split": split, "index": i},
            }

            all_conversations.append(new_conversation)

    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(all_conversations)
    df.to_json(
        output_path.replace("parquet", "jsonl"),
        index=False,
        orient="records",
        lines=True,
    )
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"Saved {len(all_conversations)} conversations to {output_path}")


def save_secrets(secrets: list, output_path: str):
    """
    Saves the list of secrets to a txt file, one secret per line.
    Args:
        secrets (list): List of secrets to save.
        output_path (str): Path to save the secrets file.
    """
    with open(output_path, "w") as f:
        for secret in secrets:
            f.write(f"{secret}\n")


def save_data(data: list, output_path: str):
    """
    Saves the list of data to a Parquet file.
    Args:
        data (list): List of data to save.
        output_path (str): Path to save the data file.
    """
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False, engine="pyarrow")


def load_inference_data(paths: list[str]) -> list[dict]:
    """
    Load multiple JSONL files and return a combined list of dictionaries.

    Args:
        paths (list[str]): List of paths to the JSONL files.

    Returns:
        list[dict]: Combined list of dictionaries from all JSONL files.
    """
    data = []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    return data


def load_jsonl(jsonl_path: str):
    """
    Load a JSONL file and return a list of dictionaries.
    Args:
        jsonl_path (str): Path to the JSONL file.
    Returns:
        list: A list of dictionaries loaded from the JSONL file.
    """
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data
