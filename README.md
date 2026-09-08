# Investigation into multiplication errors in LLMs


## Overview

Large language models frequently fail at multi-digit arithmetic, but the mechanisms behind these failures remain poorly understood. This repository contains the code used to evaluate multiplication capabilities across several small open-weight LLMs and to analyze the resulting error patterns.

We investigate whether the "noisy quantisation" model proposed for addition errors generalises to multiplication, and identify a novel error regime in which lower-order digit errors preferentially preserve parity while higher-order digit errors shift toward off-by-one carry failures.

## Repository structure

```
.
├── Collect+categorise_multiplication/        # Dataset construction (operand pair sampling) + model inference + analysis
├── additional_figures/               # Additional analysis (carry potential etc.) + figures
├── analysis/                 # Carry potential, parity preservation, and accuracy analysis
└── requirements.txt
```

## Dataset

We construct a dataset of 16,991 multiplication problems with operand sizes ranging from 1-digit×1-digit to 3-digit×4-digit. The three smallest tiers are exhaustively enumerated; the four largest are uniformly sampled without replacement. See the paper for full details of the sampling procedure and per-tier counts.

## Models evaluated

All models are evaluated in their base (non-instruction-tuned) form, with a minimal prompt format (`x1 * x2 = `):

- Qwen3.5-2B
- Qwen3.5-0.8B
- Qwen2.5-Math-1.5B
- Llama-3.2-3B
- Gemma-4-E2B

## Setup

```bash
pip install -r requirements.txt
```

Core dependencies: PyTorch, Hugging Face Transformers, NumPy, pandas, SciPy, Matplotlib.

## Usage

1. **Generate the dataset**: run the notebook cells in `Collect+categorise_multiplication/` to construct the operand-pair dataset (seeded for reproducibility).
2. **Run inference**: run the following cells in `Collect+categorise_multiplication/` to collect model answers for each model checkpoint. Results are cached as `.npy` arrays (`model_answers`, `ground_truths`, `categories`, `correct`, `operand_pairs`) to avoid re-running generation.
3. **Reproduce analysis and figures**: run the later cells in `Collect+categorise_multiplication/` and `additional_figures/` using the saved arrays.

