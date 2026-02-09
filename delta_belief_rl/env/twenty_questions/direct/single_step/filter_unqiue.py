import datasets
import numpy as np
import os

def filter_unique():
    parquet_file = 'multiturn_grpo_best_no_skip_test_gemini.parquet'
    # path
    system_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(system_path, parquet_file)

    dataframe = datasets.load_dataset("parquet", data_files=path)["train"]

    seen = set()
    def is_first(example):
        gt = example["ground_truth"]
        if gt in seen:
            return False
        seen.add(gt)
        return True

    ds_unique = dataframe.filter(is_first)
    ds_unique = ds_unique.to_pandas()
    #format to verl format 
    ds_unique["golden_answers"] = np.array(ds_unique['ground_truth'])
    ds_unique["data_source"] = np.array(["tw_q" for _ in range(len(ds_unique))])
    ds_unique["id"] = np.array(['train' + "_" + str(i) for i in range(len(ds_unique))])
    ds_unique["ability"] = np.array(["reasoning" for _ in range(len(ds_unique))])

    reward_model = []
    for i in range(len(ds_unique)):
        reward_model.append({
            "style": "rule",
            "ground_truth": {"target": [ds_unique["ground_truth"][i]]}
        })

    ds_unique["reward_model"] = np.array(reward_model)

    # print("ds_unique", ds_unique)

    ds_unique = datasets.Dataset.from_pandas(ds_unique)

    ds_unique = ds_unique.remove_columns(["ground_truth"])
    
    print("ds_unique", ds_unique)
    
    #save the filtered dataset
    output_path = os.path.join(system_path, "multiturn_grpo_best_no_skip_test_gemini_unique.parquet")
    ds_unique.to_parquet(output_path)


    print("Filtered dataset saved as 'multiturn_grpo_best_no_skip_test_gemini_unique.parquet'.")

if __name__ == "__main__":
    filter_unique()