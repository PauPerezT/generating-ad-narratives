# Generating AD Narratives

This repository packages the training, evaluation, and embedding workflows used for the AD narrative generation paper into a reproducible structure with standardized parameters across model families.

## Paper Visuals

### Main Results Snapshot

![Main paper results](assets/figures/radar_plot.svg)

### Methodology / Pipeline

![Methodology pipeline](assets/figures/methodology.svg)

## What Changed

- Removed hard-coded local paths.
- Removed embedded `wandb` API keys.
- Standardized CLI arguments across `gpt2`, `t5`, `flan-t5`, `llama2`, `mistral`, `qwen3`, and `deepseek-r1-qwen`.
- Centralized training and generation defaults in one config file.
- Standardized output locations under `artifacts/`.

## Repository Layout

- `configs/experiments.json`: shared defaults and model registry
- `scripts/train.py`: fine-tuning entry point
- `scripts/train_chat_lora.py`: chat-style LoRA fine-tuning for LLaMA, Mistral, Qwen, and DeepSeek-style runs
- `scripts/evaluate.py`: generation and metric export
- `scripts/embed_texts.py`: embedding extraction for generated or ground-truth text
- `docs/model_coverage.md`: paper-to-code coverage map
- `colab/finetune_chat_models.ipynb`: Colab workflow for chat-style model training
- `src/ad_narratives/`: shared utilities

## Expected Data Layout

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

If you want to use the `all_tog` variant from the original code, point `--data-dir` to that directory instead.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py --family flan-t5 --size base --group AD --data-dir ./data
python scripts/train.py --family t5 --size large --group HC --data-dir ./data/all_tog
python scripts/train.py --family gpt2 --size small --group AD --data-dir ./data
```

## Train Chat Models

Use this path for the paper's chat/instruction families and the extra DeepSeek-style notebook workflow.

```bash
python scripts/train_chat_lora.py --family llama2 --size 7b --group AD --data-dir ./data
python scripts/train_chat_lora.py --family mistral --size 7b --group HC --data-dir ./data
python scripts/train_chat_lora.py --family qwen3 --size 8b --group AD --data-dir ./data
python scripts/train_chat_lora.py --family deepseek-r1-qwen --size 8b --group HC --data-dir ./data
```

For gated checkpoints such as LLaMA-2, export `HF_TOKEN` before running.

## Evaluate

```bash
python scripts/evaluate.py --family flan-t5 --size base --group AD --data-dir ./data --combine-splits
python scripts/evaluate.py --family mistral --size 7b --group AD --data-dir ./data --combine-splits
```

## Embeddings

```bash
python scripts/embed_texts.py --input artifacts/predictions/flan-t5_base_AD_predictions.csv --text-column generated_answer --label generated
python scripts/embed_texts.py --input artifacts/groundtruth/Groundtruth.csv --text-column sentence --label groundtruth
```

## Notes

- `gpt2` size mapping is standardized as `small -> gpt2`, `base -> gpt2-medium`, `large -> gpt2-large`.
- The paper model set is documented in `docs/model_coverage.md`.
- The Colab-ready workflow is in `colab/finetune_chat_models.ipynb`.
- All families now share the same generation defaults unless explicitly overridden.
- Training defaults are centralized; family-specific overrides are kept minimal and transparent.
