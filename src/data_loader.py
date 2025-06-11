# src/data_loader.py

from datasets import load_dataset


class DataLoader:
    """Loads and formats the mathematical reasoning dataset (GSM8K)."""
    def load_math_data(self, subset_size=3, split="train", seed=15):
        """Loads a subset of the GSM8K dataset."""
        dataset = load_dataset("gsm8k", "main", split=split)
        return dataset.shuffle(seed=seed).select(range(min(subset_size, len(dataset))))

    def format_prompt(self, question):
        """Formats a question into the expected prompt structure."""
        return f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <reasoning> and </reasoning> tags, and <answer> and </answer> tags, respectively.\nUser: {question}\nAssistant:"
