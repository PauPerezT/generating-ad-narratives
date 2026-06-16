# Generating AD Narratives

<p align="center">
  Synthetic narrative generation for Alzheimer's disease research using GPT-2, T5, Flan-T5, LLaMA-2, Mistral, Qwen3, and a standardized DeepSeek/Qwen chat workflow.
</p>

<p align="center">
  <strong>Human-to-Bot</strong> and <strong>Bot-to-Bot</strong> simulation, standardized parameters, automatic evaluation, embeddings, and a Colab path for large chat models.
</p>

## At a Glance

This repository turns the paper code into a cleaner and reproducible project structure. The original scripts and notebooks were aligned so all model families share the same argument style, centralized defaults, and output layout.

- Dataset: DementiaBank Pitt Corpus Cookie Theft interactions
- Task: generate AD and HC narratives from interviewer-participant QA pairs
- Scenarios: `Human-to-Bot` and `Bot-to-Bot`
- Downstream use: synthetic data augmentation for text-based AD detection
- Coverage: classic baselines, instruction models, chat LoRA models, and Colab workflows

## Methodology

The pipeline leads the page because it is the core story of the paper.

![Methodology pipeline](assets/figures/methodology.svg)

The framework fine-tunes:

- an answer-generation model that maps interviewer questions to participant responses
- a question-generation model that maps participant responses back to interviewer-style prompts

Those models are then evaluated in two settings:

- `Human-to-Bot`: original corpus questions are used to generate narratives
- `Bot-to-Bot`: one model asks and the other answers to simulate full interactions

## Key Results

![Main paper results](assets/figures/radar_plot.svg)

Highlights pulled from the manuscript:

- Mistral, LLaMA, and Qwen were the strongest generation models across semantic and lexical metrics.
- In `Human-to-Bot`, Mistral 7B achieved the strongest AD generation scores with `BLEU = 0.440`, `SemScore = 0.773`, and `BERTScore = 0.909`.
- In downstream AD detection, the best result came from `Mistral 7B` in `H2B Mix2GT` with `F1 = 0.843`.
- A classifier trained only on original narratives reported `F1 = 0.72`, showing the practical value of synthetic augmentation.
- Human evaluation also ranked Mistral highest overall, with strong fluency, plausibility, and diagnostic realism.

## Supported Models

| Family | Sizes | Entry point | Status |
|---|---|---|---|
| GPT-2 | `small`, `base`, `large` | `scripts/train.py` | Included |
| T5 | `small`, `base`, `large` | `scripts/train.py` | Included |
| Flan-T5 | `small`, `base`, `large` | `scripts/train.py` | Included |
| LLaMA-2 | `7b` | `scripts/train_chat_lora.py` | Included |
| Mistral | `7b` | `scripts/train_chat_lora.py` | Included |
| Qwen3 | `8b` | `scripts/train_chat_lora.py` | Included |
| DeepSeek-R1-Qwen | `8b` | `scripts/train_chat_lora.py` | Experimental workflow included |

The model-to-paper mapping and source provenance are documented in `docs/model_coverage.md`.

## What Was Standardized

- Removed hard-coded local paths
- Removed embedded secrets such as `wandb` and Hugging Face credentials
- Standardized CLI arguments across all model families
- Centralized defaults in `configs/experiments.json`
- Standardized output locations under `artifacts/`
- Added one Colab notebook path for the shared chat-style workflows

## Repository Layout

- `configs/experiments.json`: shared defaults, generation settings, and model registry
- `scripts/train.py`: training for GPT-2, T5, and Flan-T5
- `scripts/train_chat_lora.py`: LoRA fine-tuning for LLaMA-2, Mistral, Qwen3, and DeepSeek-style runs
- `scripts/evaluate.py`: generation and automatic metrics
- `scripts/embed_texts.py`: embedding extraction for generated or ground-truth text
- `docs/model_coverage.md`: paper-to-code coverage map
- `colab/finetune_chat_models.ipynb`: Colab workflow for large chat-style models
- `src/ad_narratives/`: shared utilities

## Data Format

The scripts expect CSV files with `question` and `answer` columns.

```text
data/
  train_QA_AD_data.csv
  val_QA_AD_data.csv
  test_QA_AD_data.csv
  train_QA_HC_data.csv
  val_QA_HC_data.csv
  test_QA_HC_data.csv
```

If you want to use the original `all_tog` variant, point `--data-dir` to that directory instead.

## How To Use

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Train classic baselines

```bash
python scripts/train.py --family gpt2 --size small --group AD --data-dir ./data
python scripts/train.py --family t5 --size large --group HC --data-dir ./data
python scripts/train.py --family flan-t5 --size base --group AD --data-dir ./data
```

### 3. Train chat-style models

```bash
python scripts/train_chat_lora.py --family llama2 --size 7b --group AD --data-dir ./data
python scripts/train_chat_lora.py --family mistral --size 7b --group HC --data-dir ./data
python scripts/train_chat_lora.py --family qwen3 --size 8b --group AD --data-dir ./data
python scripts/train_chat_lora.py --family deepseek-r1-qwen --size 8b --group HC --data-dir ./data
```

For gated checkpoints such as LLaMA-2, export `HF_TOKEN` before running.

### 4. Evaluate generations

```bash
python scripts/evaluate.py --family flan-t5 --size base --group AD --data-dir ./data --combine-splits
python scripts/evaluate.py --family mistral --size 7b --group AD --data-dir ./data --combine-splits
```

### 5. Extract embeddings

```bash
python scripts/embed_texts.py --input artifacts/predictions/flan-t5_base_AD_predictions.csv --text-column generated_answer --label generated
python scripts/embed_texts.py --input artifacts/groundtruth/Groundtruth.csv --text-column sentence --label groundtruth
```

### 6. Use the Colab workflow

Open `colab/finetune_chat_models.ipynb` for the notebook version of the shared Mistral/Qwen-style training path.

## Notes

- `gpt2` uses the standardized mapping `small -> gpt2`, `base -> gpt2-medium`, `large -> gpt2-large`.
- Chat models use shared LoRA defaults with minimal family-specific overrides.
- The standardized configuration uses `seed = 42`, shared generation defaults, and centralized training hyperparameters.
- The repository focuses on reproducibility and clearer reuse of the paper workflows rather than reproducing every original local environment assumption.
