# src/baseline_tester.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data_loader import DataLoader
from src.evaluator import Evaluator


class BaselineTester:
    """Handles evaluation of the baseline model."""
    def __init__(self, data_loader: DataLoader, evaluator: Evaluator,
                 model_name: str, torch_dtype: torch.dtype, eval_config: dict):
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.eval_config = eval_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_model_and_tokenizer(self):
        """Loads and configures the model and tokenizer for baseline testing."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]", "additional_special_tokens": ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]})
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.torch_dtype)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(self.device)
        return model, tokenizer

    def run_test(self, eval_subset_size: int, run_seed: int, max_completion_length: int):
        """Executes a single baseline evaluation run."""
        print(f"Loading evaluation data for baseline run with seed {run_seed}...")
        eval_dataset = self.data_loader.load_math_data(subset_size=eval_subset_size, split="test", seed=run_seed)
        eval_dataset = eval_dataset.map(lambda x: {"prompt": x["question"], "ground_truth": x["answer"]})  # No prompt formatting for raw baseline
        eval_dataset = eval_dataset.remove_columns(["question", "answer"])
        print(f"Evaluation dataset size: {len(eval_dataset)}")

        model, tokenizer = self._setup_model_and_tokenizer()

        print("\nEvaluating baseline model...")
        baseline_pass_at_1 = self.evaluator.evaluate_pass_at_1(
            model, tokenizer, eval_dataset,
            num_samples=self.eval_config['num_samples_per_eval'],
            enforce_format=self.eval_config['enforce_format_for_baseline'],
            max_new_tokens=max_completion_length
        )
        print(f"Baseline Pass@1 Score for current run: {baseline_pass_at_1:.4f}")

        del model  # Free memory
        torch.cuda.empty_cache()
        return baseline_pass_at_1
