# src/reward_calculator.py

import re


class MathRewardCalculator:
    """Calculates rewards for generated completions based on format and accuracy."""
    def __init__(self):
        # Stores reward histories across multiple runs
        self.all_reward_histories = []
        self.all_format_reward_histories = []
        self.all_accuracy_reward_histories = []
        self.all_completion_length_histories = []

    def _extract_final_number(self, text):
        """Extracts the last numerical value from a text."""
        nums = re.findall(r'-?\d*\.?\d+', text)
        return float(nums[-1]) if nums else None

    def _extract_final_numbers(self, text, n=2):
        """Extracts the last 'n' numerical values from a text."""
        nums = re.findall(r'-?\d*\.?\d+', text)
        if len(nums) < n:
            return None
        return [float(nums[-i]) for i in range(1, n + 1)]

    def _extract_answer_from_tags(self, completion):
        """Extracts content within <answer> tags."""
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _check_format_compliance(self, completion):
        """Checks if completion adheres to expected tag format and order."""
        return (completion.count("<reasoning>") == 1 and completion.count("</reasoning>") == 1 and
                completion.count("<answer>") == 1 and completion.count("</answer>") == 1 and
                completion.find("</reasoning>") < completion.find("<answer>"))

    def calculate_reward(self, prompts, completions, ground_truth, run_idx):
        """Calculates combined format and accuracy rewards for a batch."""
        while run_idx >= len(self.all_reward_histories):
            self.all_reward_histories.append([])
            self.all_format_reward_histories.append([])
            self.all_accuracy_reward_histories.append([])
            self.all_completion_length_histories.append([])

        rewards, batch_format_rewards, batch_accuracy_rewards, batch_completion_lengths = [], [], [], []

        for prompt, completion, answer in zip(prompts, completions, ground_truth):
            format_reward, accuracy_reward = 0.0, 0.0
            batch_completion_lengths.append(len(completion.split()))

            if self._check_format_compliance(completion):
                format_reward = 0.4
                extracted_answer = self._extract_answer_from_tags(completion)
                all_numbers = [float(n) for n in re.findall(r'-?\d*\.?\d+', extracted_answer)]
                true_num = self._extract_final_number(answer)

                if all_numbers and true_num is not None and true_num in all_numbers:
                    accuracy_reward = 0.6

            total_reward = format_reward + accuracy_reward
            rewards.append(total_reward)
            batch_format_rewards.append(format_reward)
            batch_accuracy_rewards.append(accuracy_reward)

        self.all_reward_histories[run_idx].extend(rewards)
        self.all_format_reward_histories[run_idx].extend(batch_format_rewards)
        self.all_accuracy_reward_histories[run_idx].extend(batch_accuracy_rewards)
        self.all_completion_length_histories[run_idx].extend(batch_completion_lengths)

        return rewards
