# src/grpo_trainer_wrapper.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from src.data_loader import DataLoader
from src.reward_calculator import MathRewardCalculator
from src.evaluator import Evaluator


class GRPOTrainerWrapper:
    """Manages the GRPO training process."""
    def __init__(self, data_loader: DataLoader, reward_calculator: MathRewardCalculator,
                 evaluator: Evaluator, model_name: str, torch_dtype: torch.dtype,
                 training_args: dict, eval_config: dict):
        self.data_loader = data_loader
        self.reward_calculator = reward_calculator
        self.evaluator = evaluator
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.training_args_config = training_args
        self.eval_config = eval_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_model_and_tokenizer(self):
        """Loads and configures the model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]", "additional_special_tokens": ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]})
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.torch_dtype)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(self.device)
        return model, tokenizer

    def run_train(self, run_idx: int, train_subset_size: int, eval_subset_size: int, run_seed: int):
        """Executes a single GRPO training run."""
        print(f"Loading training data for run {run_idx + 1} with seed {run_seed}...")
        train_dataset = self.data_loader.load_math_data(subset_size=train_subset_size, split="train", seed=run_seed)
        train_dataset = train_dataset.map(lambda x: {"prompt": self.data_loader.format_prompt(x["question"]), "ground_truth": x["answer"]})
        train_dataset = train_dataset.remove_columns(["question", "answer"])
        print(f"Training dataset size: {len(train_dataset)}")

        print("Loading evaluation data...")
        eval_dataset = self.data_loader.load_math_data(subset_size=eval_subset_size, split="test", seed=self.eval_config['evaluation_seed'])
        eval_dataset = eval_dataset.map(lambda x: {"prompt": self.data_loader.format_prompt(x["question"]), "ground_truth": x["answer"]})
        eval_dataset = eval_dataset.remove_columns(["question", "answer"])
        print(f"Evaluation dataset size: {len(eval_dataset)}")

        model, tokenizer = self._setup_model_and_tokenizer()
        grpo_config = GRPOConfig(**self.training_args_config)

        # Custom reward function to pass the current run_idx
        def custom_reward_function(prompts, completions, ground_truth, **kwargs):
            return self.reward_calculator.calculate_reward(prompts, completions, ground_truth, run_idx=run_idx)

        trainer = GRPOTrainer(
            model=model, reward_funcs=custom_reward_function, args=grpo_config,
            train_dataset=train_dataset, processing_class=tokenizer
        )

        print("\nStarting Training...")
        trainer.train()
        print("\nTraining Completed.")

        print("\nEvaluating trained model...")
        pass_at_1 = self.evaluator.evaluate_pass_at_1(
            model, tokenizer, eval_dataset,
            num_samples=self.eval_config['num_samples_per_eval'],
            enforce_format=self.eval_config['enforce_format_for_grpo'],
            max_new_tokens=self.eval_config['max_completion_length_eval'],
            temperature=self.eval_config['temperature'],
            top_p=self.eval_config['top_p']
        )
        print(f"Pass@1 Score for current run: {pass_at_1:.4f}")

        del model, trainer  # Free memory
        torch.cuda.empty_cache()
        return pass_at_1
