# GR-PSPO

A new approach to trust region optimisation for fine-tuning language models, introducing Probability Smoothing as an alternative to traditional clipping. This repository contains the implementation of PSPO in GRPO (GR-PSPO) and scripts to run the fine-tuning on maths reasoning tasks.

## Overview

Traditional trust region methods in reinforcement learning often rely on hard clipping to constrain policy updates, which can lead to suboptimal performance. This implementation presents an alternative approach that:

1. Replaces hard clipping with soft probability interpolation
2. Preserves the beneficial properties of trust regions without the limitations of clipping
3. Provides more stable training for language model fine-tuning

We compare three TRL methods for comparison:
- GRPO without clipping
- GRPO with clipping
- GR-PSPO (GRPO with our proposed method) - this is implemented in the patches for the TRL GRPOTrainer and Configs (in grpotrainer.patch and grpoconfig.patch).

## Implementation Details

### Training System
- `maths_finetuning.py`: Main training loop, fine-tuning a base model on the GSM8K training set.
- `grpotrainer|config.patch`: GR-PSPO as a wrapper of the GRPOTrainer and Config, where the trust region method decides if it is clipping (='clip') or PSPO (='pspo').
- `maths_rewards.py`: Reward function set up and response parsing for mathematical reasoning tasks

### Analysis Tools
- `save_model_locally.py`: Model and dataset management - to use iridis needed to have all the data offline, so this script saves the datasets and models to a local directory cached_files

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download models and datasets:
```bash
python save_model_locally.py
```

This will setup:
- Qwen models (0.5B, 1.5B, 3B variants)
- Mathematical reasoning datasets:
  - GSM8K
  - MATH-500
  - ASDIV
  - SVAMP

## Usage

### Training

Train a model using different trust region methods:

```bash
python maths_finetuning.py --model [model_name] --trust_region_method [method] --seed [seed]
```

Available methods:
- `clip`: PPO with standard clipping
- `noclip`: GRPO without clipping
- `pspo`: Our PSPO method
- `sft`: Supervised fine-tuning baseline

### Evaluation

Evaluate model performance:

```bash
python maths_evaluations.py --model_path [trained_model_path] --samples_per_prompt [n]
```

## Citation

```bibtex
@article{dwyer2025notclipping,
  title={It's Not You, It's Clipping: A Soft Trust-Region via Probability Smoothing for LLM RL},
  author={Dwyer, Madeleine and Sobey, Adam and Chapman, Adriane},
  journal={arXiv preprint arXiv:2509.21282},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
