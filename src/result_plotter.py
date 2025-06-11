# src/result_plotter.py

import numpy as np
import matplotlib.pyplot as plt


class ResultPlotter:
    """Saves experiment results and plots combined reward histories."""
    def __init__(self, plot_data_filename="grpo_training_results.npz",
                 combined_plot_filename="combined_rewards_and_scores.png"):
        self.plot_data_filename = plot_data_filename
        self.combined_plot_filename = combined_plot_filename

    def save_plot_data(self, reward_history, format_reward_history,
                       accuracy_reward_history, completion_length_history,
                       pass_at_1_scores):
        """Saves all reward histories and evaluation scores to a numpy file."""
        np.savez(
            self.plot_data_filename,
            reward_histories=np.array(reward_history, dtype=object),
            format_reward_histories=np.array(format_reward_history, dtype=object),
            accuracy_reward_histories=np.array(accuracy_reward_history, dtype=object),
            completion_length_histories=np.array(completion_length_history, dtype=object),
            pass_at_1_scores=np.array(pass_at_1_scores, dtype=object)
        )
        print(f"Plot data saved to {self.plot_data_filename}")

    def plot_combined_rewards(self):
        """Loads data from the configured filename and generates a combined plot."""
        try:
            data = np.load(self.plot_data_filename, allow_pickle=True)
            reward_histories = np.vstack([h.astype(float) for h in data['reward_histories']])
            format_histories = np.vstack([h.astype(float) for h in data['format_reward_histories']])
            accuracy_histories = np.vstack([h.astype(float) for h in data['accuracy_reward_histories']])
            completion_lengths = np.vstack([h.astype(float) for h in data['completion_length_histories']])
            pass_at_1_scores = data['pass_at_1_scores']

        except FileNotFoundError:
            print(f"Error: Plot data file not found at {self.plot_data_filename}. Cannot generate plot.")
            return
        except Exception as e:
            print(f"Error loading or processing data from {self.plot_data_filename}: {e}. Cannot generate plot.")
            return

        def calculate_stats(arr):
            """Calculates mean and standard error of the mean for an array."""
            mean = np.mean(arr, axis=0)
            std_err = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
            return mean, std_err

        def smooth_data(x, window=41):
            """Applies a simple moving average for smoothing."""
            if len(x) < window: return x
            kern = np.ones(window) / window
            pad = np.pad(x, (window // 2, window // 2), mode='reflect')
            return np.convolve(pad, kern, mode='valid')

        metrics_to_plot = [
            (reward_histories, 'Total Reward', (-0.05, 1.05), 1.0),
            (format_histories, 'Format Reward', (-0.05, 1.05), 0.4),
            (accuracy_histories, 'Correct Answer Reward', (-0.05, 1.05), 0.6),
            (completion_lengths, 'Completion Length', (0, None), None)
        ]

        plt.style.use('ggplot')
        fig, axes = plt.subplots(nrows=len(metrics_to_plot), ncols=1, figsize=(10, 5 * len(metrics_to_plot)), sharex=True)
        if len(metrics_to_plot) == 1: axes = np.expand_dims(axes, 0) # Ensure 2D indexing for single metric

        for idx, (data_arr, title, ylim, target_line) in enumerate(metrics_to_plot):
            mean, se = calculate_stats(data_arr)
            smoothed_mean = smooth_data(mean)
            smoothed_se = smooth_data(se)
            steps = np.arange(len(smoothed_mean))

            ax = axes[idx]
            ax.plot(steps, smoothed_mean, label='Mean', linewidth=2.5, color='#1A5276')
            ax.fill_between(steps, smoothed_mean - smoothed_se, smoothed_mean + smoothed_se, color='#1A5276', alpha=0.25, label='Std Error')

            if target_line is not None:
                ax.axhline(y=target_line, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7, label='Optimum Reward')

            ax.set_title(title, fontsize=18, fontweight='bold')
            ax.set_ylabel(title, fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)

        axes[-1].set_xlabel('Training Steps', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.combined_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved combined plot to {self.combined_plot_filename}")

        if pass_at_1_scores.size > 0:
            mean_pass_at_1 = np.mean(pass_at_1_scores)
            std_pass_at_1 = np.std(pass_at_1_scores)
            print(f"Final Pass@1 Score (Mean ± Std): {mean_pass_at_1:.4f} ± {std_pass_at_1:.4f}")
