

#load txt file 
import os

cwd = os.getcwd()
filename_test_gemini = os.path.join(cwd, 'delta_belief_rl/env/twenty_questions/config/gemini/test_secrets.txt')
filename_val_gemini = os.path.join(cwd, 'delta_belief_rl/env/twenty_questions/config/gemini/val_secrets.txt')
filename_coca_sft = os.path.join(cwd, 'delta_belief_rl/env/twenty_questions/config/coca/sft_single_turn_training.txt')


with open(filename_test_gemini, 'r') as file:
    test_gemini = file.read().splitlines()

with open(filename_val_gemini, 'r') as file:
    val_gemini = file.read().splitlines()

with open(filename_coca_sft, 'r') as file:
    sft_coca = file.read().splitlines()


# Check which words are overlapping
test_gemini = set(test_gemini)
val_gemini = set(val_gemini)
sft_coca = set(sft_coca)
overlapping_test = test_gemini.intersection(sft_coca)
print(f'Number of overlapping test words: {len(overlapping_test)}')

overlapping_val = val_gemini.intersection(sft_coca)
print(f'Number of overlapping val words: {len(overlapping_val)}')

# Remove overlapping words from test and val gemini 
unique_gemini_test = test_gemini - overlapping_test
print(f'Number of gemini test words after removing SFT overlaps: {len(unique_gemini_test)/len(test_gemini):.2%} of original')
print(f'Unique gemini test words: {len(unique_gemini_test)}')

unique_gemini_val = val_gemini - overlapping_val
print(f'Number of gemini val words after removing SFT overlaps: {len(unique_gemini_val)/len(val_gemini):.2%} of original')
print(f'Unique gemini val words: {len(unique_gemini_val)}')

# Save to a new txt file 
filename_output = os.path.join(cwd, 'delta_belief_rl/env/twenty_questions/config/coca_plus/unique_gemini_test.txt')
with open(filename_output, 'w') as f:
    for item in unique_gemini_test:
        f.write("%s\n" % item)

filename_output = os.path.join(cwd, 'delta_belief_rl/env/twenty_questions/config/coca_plus/unique_gemini_val.txt')
with open(filename_output, 'w') as f:
    for item in unique_gemini_val:
        f.write("%s\n" % item)


# # Load all secrets
# words = []
# old_secrets = set()

# # old_rl_train_path = "delta_belief_rl/env/twenty_questions/config/coca/rl_training.txt"
# # with open(old_rl_train_path, "r") as f:
# #     words = [word.strip() for word in f.readlines()]
# #     old_secrets.update(words)
# # print("Total number of coca RL secrets:", len(old_secrets))
# old_sft_train_path = os.path.join(cwd, 'delta_belief_rl/env/twenty_questions/config/coca/sft_single_turn_training.txt')
# with open(old_sft_train_path, "r") as f:
#     words = [word.strip() for word in f.readlines()]
#     old_secrets.update(words)
# print("Total number of coca SFT secrets:", len(old_secrets))
# # old_train_path = "delta_belief_rl/env/twenty_questions/config/coca/train.txt"
# # with open(old_train_path, "r") as f:
# #     words = [word.strip() for word in f.readlines()]
# #     old_secrets.update(words)
# # print("Total number of coca train secrets:", len(old_secrets))

# # Load test secrets
# test_secrets = set()
# test_path = os.path.join(cwd, 'delta_belief_rl/env/twenty_questions/config/gemini/test_secrets.txt')
# with open(test_path, "r") as f:
#     words = [word.strip() for word in f.readlines()]
#     test_secrets.update(words)
# print("Total number of test secrets:", len(test_secrets))

# # Remove sensitive secrets from all secrets
# unleaked_secrets = test_secrets - old_secrets
# print("Total number of unleaked test secrets:", len(unleaked_secrets))
# print(
#     f"Percentage overlap: {(1.0 - (len(unleaked_secrets) / len(test_secrets))) * 100}%"
# )
