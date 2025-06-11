# Generalized Reinforcement Policy Optimization for Mathematical Reasoning

This repository contains the code developed for my Bachelor thesis:  
**"No Supervision, No Problem: Pure Reinforcement Learning Improves Mathematical Reasoning in Small Language Models."**

It implements **Generalized Reinforcement Policy Optimization (GRPO)** to improve the mathematical reasoning abilities of small language models through pure reinforcement learning.  

---

## 🛠️ Usage

Edit your experiment settings in:  
`config/experiment_config.yaml`

Then run:

# GRPO training (and evaluation)
```bash
python run_experiment.py --mode train --config config/experiment_config.yaml
```

# Baseline evaluation
```bash
python run_experiment.py --mode baseline --config config/experiment_config.yaml
```

# Both training and baseline
```bash
python run_experiment.py --mode both --config config/experiment_config.yaml
```
