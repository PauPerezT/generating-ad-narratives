# Model Coverage

This repository now reflects the model families described in the paper and the additional code sources originally shared during repository assembly.

## Paper Models

| Model family | Sizes in paper | Standardized entry point | Notes |
|---|---|---|---|
| GPT-2 | `small`, `base`, `large` | `scripts/train.py` | Decoder-only QA baseline using `Question: ... Answer:` formatting |
| T5 | `small`, `base`, `large` | `scripts/train.py` | Seq2seq QA baseline |
| Flan-T5 | `small`, `base`, `large` | `scripts/train.py` | Instruction-tuned seq2seq QA baseline |
| LLaMA-2 | `7b` | `scripts/train_chat_lora.py` | Chat-style LoRA fine-tuning |
| Mistral | `7b` | `scripts/train_chat_lora.py` | Chat-style LoRA fine-tuning |
| Qwen3 | `8b` | `scripts/train_chat_lora.py` | Chat-style LoRA fine-tuning |

## Additional Shared Code

| Model family | Source context | Standardized status |
|---|---|---|
| DeepSeek | Shared source folder and Qwen/DeepSeek Colab notebook | Included as `deepseek-r1-qwen` in `scripts/train_chat_lora.py` for experimental chat-style training |
| Colab / Unsloth workflows | Shared Qwen and Mistral notebooks | Standardized Colab notebook added under `colab/` |

## Source Provenance

The repository structure was aligned against these source files and notebooks:

- `1_finetuning_gpt2.py`
- `1_finetuning_t5.py`
- `1_finetuning_flan_t5.py`
- `1_finetuning_flant5.py`
- `1_finetuning_flant5_HC.py`
- `1_finetuning_llama.py`
- `finetuning_llama2.py`
- `1_finetuning_Mistral.py`
- `1_finetuning_deepseek.py`
- `ChatML + chat templates + Qwen full example.ipynb`
- `Another copy of Copy of ChatML + chat templates + Mistral 7b full example.ipynb`

## Security Cleanup

Original shared scripts contained hard-coded private credentials for Weights & Biases and Hugging Face. The standardized repo intentionally removes those secrets and expects credentials via environment variables such as `HF_TOKEN` and normal `wandb login` workflows.
