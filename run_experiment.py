# run_experiment.py

import argparse
import yaml
import numpy as np
import torch
from src.data_loader import DataLoader
from src.reward_calculator import MathRewardCalculator
from src.evaluator import Evaluator
from src.grpo_trainer_wrapper import GRPOTrainerWrapper
from src.baseline_tester import BaselineTester
from src.result_plotter import ResultPlotter


def main():
    """Main function to parse arguments, load config, and run experiments."""
    parser = argparse.ArgumentParser(description="Run GRPO training, baseline testing, or both.")
    parser.add_argument("--mode", choices=["train", "baseline", "both"], default="train",
                        help="Mode to run: 'train' for GRPO, 'baseline' for baseline, 'both' for both.")
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml",
                        help="Path to the experiment configuration YAML file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configurations
    num_runs = config['experiment']['runs']
    train_subset_size = config['experiment']['train_size']
    eval_subset_size = config['experiment']['eval_size']
    seed_offset = config['experiment']['seed_offset']
    model_name = config['model']['name']
    torch_dtype = getattr(torch, config['model']['torch_dtype'])
    training_args = config['training_args']
    eval_config = config['evaluation']
    paths = config['paths']

    all_grpo_pass_at_1_scores = []
    all_baseline_pass_at_1_scores = []

    # Instantiate core components
    data_loader = DataLoader()
    reward_calculator = MathRewardCalculator()
    evaluator = Evaluator(data_loader)
    plotter = ResultPlotter(plot_data_filename=paths['plot_data_filename'], combined_plot_filename=paths['combined_plot_filename'])

    if args.mode in ["train", "both"]:
        print(f"\n===== Starting GRPO Training for {num_runs} runs =====")
        grpo_trainer = GRPOTrainerWrapper(
            data_loader=data_loader, reward_calculator=reward_calculator,
            evaluator=evaluator,
            model_name=model_name, torch_dtype=torch_dtype,
            training_args=training_args, eval_config=eval_config
        )

        for run_idx in range(num_runs):
            print(f"\n--- GRPO Run {run_idx + 1}/{num_runs} ---")
            run_pass_at_1 = grpo_trainer.run_train(
                run_idx=run_idx, train_subset_size=train_subset_size,
                eval_subset_size=eval_subset_size, run_seed=seed_offset + run_idx
            )
            all_grpo_pass_at_1_scores.append(run_pass_at_1)
            print(f"GRPO Run {run_idx + 1} Pass@1 Score: {run_pass_at_1:.4f}")

        # Save and plot results after all GRPO runs
        plotter.save_plot_data(
            reward_history=reward_calculator.all_reward_histories,
            format_reward_history=reward_calculator.all_format_reward_histories,
            accuracy_reward_history=reward_calculator.all_accuracy_reward_histories,
            completion_length_history=reward_calculator.all_completion_length_histories,
            pass_at_1_scores=np.array(all_grpo_pass_at_1_scores)
        )
        plotter.plot_combined_rewards()

        if all_grpo_pass_at_1_scores:
            mean_grpo = np.mean(all_grpo_pass_at_1_scores)
            std_grpo = np.std(all_grpo_pass_at_1_scores)
            print(f"\nAggregate GRPO Results across {num_runs} runs:")
            print(f"GRPO Pass@1 Score: {mean_grpo:.4f} ± {std_grpo:.4f}")

    if args.mode in ["baseline", "both"]:
        print(f"\n===== Starting Baseline Testing for {num_runs} runs =====")
        baseline_tester = BaselineTester(
            data_loader=data_loader, evaluator=evaluator,
            model_name=model_name, torch_dtype=torch_dtype, eval_config=eval_config
        )

        for run_idx in range(num_runs):
            print(f"\n--- Baseline Run {run_idx + 1}/{num_runs} ---")
            run_pass_at_1 = baseline_tester.run_test(
                eval_subset_size=eval_subset_size,
                run_seed=seed_offset + run_idx,
                max_completion_length=eval_config['max_completion_length_eval']
            )
            all_baseline_pass_at_1_scores.append(run_pass_at_1)
            print(f"Baseline Run {run_idx + 1} Pass@1 Score: {run_pass_at_1:.4f}")

        if all_baseline_pass_at_1_scores:
            mean_baseline = np.mean(all_baseline_pass_at_1_scores)
            std_baseline = np.std(all_baseline_pass_at_1_scores)
            print(f"\nAggregate Baseline Results across {num_runs} runs:")
            print(f"Baseline Pass@1 Score: {mean_baseline:.4f} ± {std_baseline:.4f}")

    print("\nExperiment(s) finished.")


if __name__ == "__main__":
    main()
