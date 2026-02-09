import datasets
import numpy as np
import os
import argparse

def generate_curriculum(args: argparse.Namespace):
    # path
    system_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(system_path, args.parquet_file)

    print(f"Loading dataset from {path}")
    dataframe = datasets.load_dataset("parquet", data_files=path)["train"]

    #verify initial prompt 
    print("initial prompt", dataframe[0]["prompt"][-3:])

    #remove the last entry of the prompt
    dataframe = dataframe.map(lambda x: {**x, "prompt": x["prompt"][:-1]})

    #verify prompt after modification
    print("modified prompt", dataframe[0]["prompt"][-3:])

    # Add verl-format columns directly to the HuggingFace Dataset
    dataframe = dataframe.add_column("golden_answers", dataframe["ground_truth"])
    dataframe = dataframe.add_column("data_source", ["tw_q"] * len(dataframe))
    dataframe = dataframe.add_column("id", [f"train_{i}" for i in range(len(dataframe))])
    dataframe = dataframe.add_column("ability", ["reasoning"] * len(dataframe))

    def make_reward_model(example):
        return {
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": [example["ground_truth"]]}
            }
        }

    dataframe = dataframe.map(make_reward_model)

    dataframe = dataframe.remove_columns(["ground_truth"])
    print("ds", dataframe)

    #split into train and test based on random split
    #set to a fixed seed for reproducibility
    
    train_test_split_dataset = dataframe.train_test_split(
                                        test_size=0.2,            # 20% for testing, 80% for training
                                        seed=42,                  # For reproducibility
                                        )

    train_ds = train_test_split_dataset['train']
    test_ds = train_test_split_dataset['test']

    #report dataset sizes
    print("ds_train", len(train_ds))
    print("ds_test", len(test_ds))
    
    #save the filtered dataset
    output_path_train = os.path.join(system_path, args.output_file_train)
    output_path_test = os.path.join(system_path, args.output_file_test)
    train_ds.to_parquet(output_path_train)
    test_ds.to_parquet(output_path_test)

    print(f"Filtered dataset saved as '{args.output_file_train}' and '{args.output_file_test}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_file", type=str, default="test_data_no_cot_new/gemini/multiturn_grpo_all_no_skip_test_gemini.parquet")
    parser.add_argument("--output_file_train", type=str, default="curriculum_gemini_train.parquet")
    parser.add_argument("--output_file_test", type=str, default="curriculum_gemini_test.parquet")
    args = parser.parse_args()
    generate_curriculum(args)