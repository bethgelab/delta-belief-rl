import argparse
import random
import os


def split_words(input_file, train_ratio=0.8, output_dir=None, seed=None):
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Read all words from the input file
    with open(input_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    # Shuffle the list of words
    random.shuffle(words)

    # Compute split index
    split_index = int(len(words) * train_ratio)

    # Split into training and test sets
    train_words = words[:split_index]
    test_words = words[split_index:]

    # Determine output directory
    base_dir = output_dir or os.path.dirname(os.path.abspath(input_file))
    os.makedirs(base_dir, exist_ok=True)

    # Write training words
    train_file = os.path.join(base_dir, "train.txt")
    with open(train_file, "w", encoding="utf-8") as f:
        for word in train_words:
            f.write(word + "\n")

    # Write test words
    test_file = os.path.join(base_dir, "test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        for word in test_words:
            f.write(word + "\n")

    print(f"Wrote {len(train_words)} training words to {train_file}")
    print(f"Wrote {len(test_words)} test words to {test_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a list of words into training and test sets."
    )
    parser.add_argument(
        "--input_file",
        default="delta_belief_rl/env/twenty_questions/objects.txt",
        help="Path to the input .txt file containing one word per line",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of words to use for training (default: 0.8)",
    )
    parser.add_argument(
        "--output_dir",
        default="delta_belief_rl/env/twenty_questions",
        help="Directory to write the output files (default: same as input file)",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    split_words(
        args.input_file,
        train_ratio=args.train_ratio,
        output_dir=args.output_dir,
        seed=args.seed,
    )
