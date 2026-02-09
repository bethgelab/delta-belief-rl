import datasets
import numpy as np
import os
import argparse
import json
import random
from collections import defaultdict
from datetime import datetime

def report_coverage(df: datasets.Dataset, secrets: list)->tuple[float, list]:
    # report coverage of secrets
    sorted_ground_truths = set(df['ground_truth'])
    missing_secrets = set(secrets) - sorted_ground_truths
    coverage = len(sorted_ground_truths) / len(secrets)
    missing_secrets = sorted(missing_secrets)
    return coverage, missing_secrets


def sample_test_dataset(df: datasets.Dataset, n: int = 1000):
    # Set random seed for reproducibility
    random.seed(42)

    # Group rows by 'ground_truth'
    gt_to_indices = defaultdict(list)
    for idx, gt in enumerate(df['ground_truth']):
        gt_to_indices[gt].append(idx)

    # First, select one sample for each unique 'ground_truth'
    selected_indices = []
    for indices in gt_to_indices.values():
        selected_indices.append(random.choice(indices))

    # Now, fill up to 1000 samples, sampling randomly from the remaining indices
    remaining_indices = set(range(len(df))) - set(selected_indices)
    remaining_indices = list(remaining_indices)
    random.shuffle(remaining_indices)
    num_to_sample = n - len(selected_indices)
    if num_to_sample > 0:
        selected_indices += remaining_indices[:num_to_sample]

    # Subset the filtered_df to these indices
    sampled_df = df.select(selected_indices)

    return sampled_df


def generate_curriculum(args: argparse.Namespace):
    # Prepare log dictionary
    log = {}

    #add timestamp to the log
    log["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # path
    system_path = os.path.dirname(os.path.abspath(__file__))
    path_train = os.path.join(system_path, args.parquet_file_train)
    path_test = os.path.join(system_path, args.parquet_file_test)

    log["loading_dataset_from"] = { "train" : path_train, "test" : path_test }
    dataframe_train = datasets.load_dataset("parquet", data_files=path_train)["train"]
    dataframe_test = datasets.load_dataset("parquet", data_files=path_test)["train"]
    # report original size
    log["original_size"] = { "train" : len(dataframe_train), "test" : len(dataframe_test) }

    # filter the data based on len(prompt)/2 == finished at 
    filtered_df_train = dataframe_train.filter(lambda x: (len(x['prompt']) / 2) == x['finished_at'])
    filtered_df_test = dataframe_test.filter(lambda x: (len(x['prompt']) / 2) == x['finished_at'])
    log["filtered_size"] = { "train" : len(filtered_df_train), "test" : len(filtered_df_test) }

    # split into train and test based on train and test secrets
    # load the secrets as a list
    train_secrets = open(args.train_secrets, "r").read().splitlines()
    test_secrets = open(args.test_secrets, "r").read().splitlines()
    train_ds = filtered_df_train.filter(lambda x: x['ground_truth'] in train_secrets)
    test_ds = filtered_df_test.filter(lambda x: x['ground_truth'] in test_secrets)

    # report dataset sizes
    log["ds_size"] = { "train" : len(train_ds), "test" : len(test_ds) }

    # sample 10000 for test
    test_ds = sample_test_dataset(test_ds, n=1000)
    log["ds_size"]["test_sampled"] = len(test_ds)

    #resport coverage of secrets 
    coverage_train, missing_secrets_train = report_coverage(train_ds, train_secrets)
    coverage_test, missing_secrets_test = report_coverage(test_ds, test_secrets)
    log["coverage"] = { "train" : coverage_train, "test" : coverage_test }
    log["missing_secrets"] = { "train" : missing_secrets_train, "test" : missing_secrets_test }

    # save the filtered dataset
    output_path_train = os.path.join(system_path, args.output_file_train)
    output_path_test = os.path.join(system_path, args.output_file_test)
    train_ds.to_parquet(output_path_train)
    test_ds.to_parquet(output_path_test)

    
    # save the filtered dataset
    output_path_train = os.path.join(system_path, args.output_file_train)
    output_path_test = os.path.join(system_path, args.output_file_test)
    train_ds.to_parquet(output_path_train)
    test_ds.to_parquet(output_path_test)

    log["filtered_dataset_saved_as"] = {
        "train": args.output_file_train,
        "test": args.output_file_test
    }

    #make dir if not exists
    os.makedirs(os.path.join("../../config/curriculum"), exist_ok=True)

    # Write log to JSON file
    log_path = os.path.join("../../config/curriculum/curriculum_generation_log.json") 
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_file_train", type=str, default="singleturn_grpo_all_easy.parquet")
    parser.add_argument("--parquet_file_test", type=str, default="singleturn_grpo_all_no_skip_test_all_models.parquet")
    parser.add_argument("--train_secrets", type=str, default="../../config/coca/train.txt")
    parser.add_argument("--test_secrets", type=str, default="../../config/coca/test.txt")
    parser.add_argument("--output_file_train", type=str, default="curriculum_train.parquet")
    parser.add_argument("--output_file_test", type=str, default="curriculum_test.parquet")
    args = parser.parse_args()
    generate_curriculum(args)