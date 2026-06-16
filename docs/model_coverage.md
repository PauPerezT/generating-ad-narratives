# Model Coverage

This document maps the paper's model families and shared notebooks to the code included in this repository.

## Paper Models

| Model family | Sizes in paper | Entry point | Notes |
|---|---|---|---|
| GPT-2 | `small`, `base`, `large` | `scripts/train.py` | Decoder-only QA baseline using `Question: ... Answer:` formatting |
| T5 | `small`, `base`, `large` | `scripts/train.py` | Seq2seq QA baseline |
| Flan-T5 | `small`, `base`, `large` | `scripts/train.py` | Instruction-tuned seq2seq QA baseline |
| LLaMA-2 | `7b` | `scripts/train_chat_lora.py` | Chat-style LoRA fine-tuning |
| Mistral | `7b` | `scripts/train_chat_lora.py` | Chat-style LoRA fine-tuning |
| Qwen3 | `8b` | `scripts/train_chat_lora.py` | Chat-style LoRA fine-tuning |

## Additional Included Workflows

| Model family | Source context | Repository implementation |
|---|---|---|
| Colab / Unsloth workflows | Shared Qwen and Mistral notebooks | Included in `colab/finetune_chat_models.ipynb` |

## Source Files Reflected In This Repository

- `1_finetuning_gpt2.py`
- `1_finetuning_t5.py`
- `1_finetuning_flan_t5.py`
- `1_finetuning_flant5.py`
- `1_finetuning_flant5_HC.py`
- `1_finetuning_llama.py`
- `finetuning_llama2.py`
- `1_finetuning_Mistral.py`
- `ChatML + chat templates + Qwen full example.ipynb`
- `Another copy of Copy of ChatML + chat templates + Mistral 7b full example.ipynb`

## Credentials

The repository expects credentials through environment variables and standard login flows such as `HF_TOKEN` and `wandb login`.
