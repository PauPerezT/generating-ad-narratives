# Generating AD Narratives

<p align="center">
  Synthetic narrative generation for Alzheimer's disease research using GPT-2, T5, Flan-T5, LLaMA-2, Mistral, and Qwen3.
</p>

<p align="center">
  <strong>Human-to-Bot</strong> and <strong>Bot-to-Bot</strong> simulation, automatic evaluation, embedding extraction, and Colab support for large chat models.
</p>

## Overview

This repository accompanies the paper on synthetic narrative generation for Alzheimer's disease detection from Cookie Theft picture-description interactions. It includes the training, generation, evaluation, and embedding workflows used to study how large language models can simulate clinically meaningful AD and HC narratives.

- Dataset: DementiaBank Pitt Corpus Cookie Theft interactions
- Task: generate AD and HC narratives from interviewer-participant QA pairs
- Scenarios: `Human-to-Bot` and `Bot-to-Bot`
- Downstream task: synthetic data augmentation for text-based AD detection
- Model families: GPT-2, T5, Flan-T5, LLaMA-2, Mistral, and Qwen3

## Methodology

![Methodology pipeline](assets/figures/methodology.svg)

The study fine-tunes two complementary generation pipelines:

- an answer-generation model that maps interviewer questions to participant responses
- a question-generation model that maps participant responses back to interviewer-style prompts

These models are evaluated in two settings:

- `Human-to-Bot`: original corpus questions are used to generate synthetic participant narratives
- `Bot-to-Bot`: one model generates questions and another model generates responses to simulate full interactions

## Main Results

<p align="center">
  <img src="assets/figures/radar_plot.svg" alt="Main paper results" width="560">
</p>

Key findings from the paper:

- Mistral, LLaMA, and Qwen were the strongest generation models across semantic and lexical metrics.
- In `Human-to-Bot`, Mistral 7B achieved the strongest AD generation scores with `BLEU = 0.440`, `SemScore = 0.773`, and `BERTScore = 0.909`.
- In downstream AD detection, the best result came from `Mistral 7B` in `H2B Mix2GT` with `F1 = 0.843`.
- A classifier trained only on original narratives reported `F1 = 0.72`, showing the value of synthetic augmentation.
- Human evaluation also ranked Mistral highest overall, with strong fluency, plausibility, and diagnostic realism.

| Result area | Best finding |
|---|---|
| Narrative generation | `Mistral 7B` delivered the strongest overall semantic and lexical performance |
| AD generation quality | `BLEU = 0.440`, `SemScore = 0.773`, `BERTScore = 0.909` in `Human-to-Bot` |
| AD detection | `Mistral 7B` reached `F1 = 0.843` in `H2B Mix2GT` |
| Baseline reference | Original-only classification achieved `F1 = 0.72` |
| Human evaluation | Mistral ranked highest in realism and diagnostic appropriateness |

## Generated Narrative Examples

Prompt used in the paper:

> What action do you see going on in that picture?

Selected examples from the manuscript:

| Model | AD-style generated narrative | HC-style generated narrative |
|---|---|---|
| GPT-2 | `mother at sink, water running over, boy on stool taking cookies` | `the boy is reaching for cookies on a stool while his sister watches and the mother washes dishes` |
| Mistral | `the boy is taking cookies and the stool may fall while water runs from the sink` | `the boy is standing on a stool reaching for cookies while his sister reaches up and the sink is overflowing as the mother dries dishes` |
| Qwen | `boy reaching for cookies on stool, water spilling from sink` | `the boy stands on a stool taking cookies while the girl reaches up and the mother washes dishes as water runs from the sink` |

These examples reflect one of the main qualitative findings of the paper: AD-style generations are often shorter and less detailed, while HC-style generations tend to integrate more of the scene and produce fuller descriptions.

## Included Models

| Family | Sizes | Entry point | Use in repository |
|---|---|---|---|
| GPT-2 | `small`, `base`, `large` | `scripts/train.py` | Decoder-only QA baseline |
| T5 | `small`, `base`, `large` | `scripts/train.py` | Seq2seq QA baseline |
| Flan-T5 | `small`, `base`, `large` | `scripts/train.py` | Instruction-tuned seq2seq QA baseline |
| LLaMA-2 | `7b` | `scripts/train_chat_lora.py` | Chat-style LoRA fine-tuning |
| Mistral | `7b` | `scripts/train_chat_lora.py` | Chat-style LoRA fine-tuning |
| Qwen3 | `8b` | `scripts/train_chat_lora.py` | Chat-style LoRA fine-tuning |
Model coverage and source mapping are documented in `docs/model_coverage.md`.

## Repository Layout

- `configs/experiments.json`: shared training, generation, and model configuration
- `scripts/train.py`: training for GPT-2, T5, and Flan-T5
- `scripts/train_chat_lora.py`: LoRA fine-tuning for LLaMA-2, Mistral, and Qwen3
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

Open `colab/finetune_chat_models.ipynb` for the notebook workflow for Mistral and Qwen experiments.

## Notes

- `gpt2` uses the mapping `small -> gpt2`, `base -> gpt2-medium`, `large -> gpt2-large`.
- Chat models use shared LoRA defaults with minimal family-specific overrides.
- The default configuration uses `seed = 42` with centralized training and generation settings in `configs/experiments.json`.
