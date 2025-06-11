# src/evaluator.py

import torch
import re
from src.reward_calculator import MathRewardCalculator


class Evaluator:
    """Evaluates the model's performance using the Pass@1 metric."""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.reward_calculator = MathRewardCalculator()

    def evaluate_pass_at_1(self, model, tokenizer, eval_dataset, num_samples=6,
                           enforce_format=True, max_new_tokens=6144,
                           temperature=0.6, top_p=0.95):
        """Calculates Pass@1 score (as average proportion of correct samples per item)."""
        model.eval()
        correct_proportions = []

        for item in eval_dataset:
            prompt = item["prompt"]
            ground_truth = item["ground_truth"]
            true_num = self.reward_calculator._extract_final_number(ground_truth)
            if true_num is None:
                continue

            correct_samples_for_item = 0
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            for i in range(num_samples):
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids, attention_mask=attention_mask, temperature=temperature, top_p=top_p,
                        do_sample=True, pad_token_id=tokenizer.pad_token_id, max_new_tokens=max_new_tokens,
                    )
                completion = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
                print("completion:", completion)

                is_correct = False
                if enforce_format:
                    if self.reward_calculator._check_format_compliance(completion):
                        extracted_answer = self.reward_calculator._extract_answer_from_tags(completion)
                        print("extracted answer:", extracted_answer)
                        gen_nums = [float(n) for n in re.findall(r'-?\d*\.?\d+', extracted_answer)]
                        is_correct = true_num in gen_nums if gen_nums else False
                else:  # For baseline, extract from raw completion
                    extracted_answer = completion
                    gen_nums = self.reward_calculator._extract_final_numbers(extracted_answer, n=2)
                    is_correct = gen_nums is not None and true_num in gen_nums

                if is_correct:
                    correct_samples_for_item += 1
                    print("! correct !")

            proportion_correct = correct_samples_for_item / num_samples
            correct_proportions.append(proportion_correct)

        pass_at_1 = sum(correct_proportions) / len(correct_proportions) if correct_proportions else 0
        return pass_at_1
