import datasets
import numpy as np
import os

def filter_unique():
    parquet_file = 'multiturn_grpo_best_no_skip_test_gemini_unique.parquet'
    # path
    system_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(system_path, parquet_file)

    dataframe = datasets.load_dataset("parquet", data_files=path)["train"]
    original = dataframe.to_pandas()

    fake_prompts = []
    for prompt in original['prompt']:
        modified_prompt = prompt.copy()
        user_ansewr = modified_prompt[-2]['content']
        if user_ansewr == 'Yes':
            modified_prompt[-2] = {'content': 'No', 'role':'user'}
        elif user_ansewr == 'No':
            modified_prompt[-2] = {'content': 'Yes', 'role':'user'}
        fake_prompts.append(modified_prompt)

    assert original['prompt'][0][-2]['content'] != fake_prompts[0][-2]['content'], "The content should be changed."

    original['prompt'] = fake_prompts

    ds_unique = datasets.Dataset.from_pandas(original)

    
    print("ds_unique", ds_unique)
    
    #save the filtered dataset
    output_path = os.path.join(system_path, "multiturn_grpo_best_no_skip_test_gemini_unique_flip.parquet")
    ds_unique.to_parquet(output_path)


    print("Filtered dataset saved as 'multiturn_grpo_best_no_skip_test_gemini_unique_flip.parquet'.")

if __name__ == "__main__":
    filter_unique()