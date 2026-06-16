import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from trl import SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ad_narratives.shared import (  # noqa: E402
    build_run_name,
    format_chat_prompt,
    get_run_paths,
    load_config,
    load_split_frames,
    resolve_family_config,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=["llama2", "mistral", "qwen3", "deepseek-r1-qwen"],
        required=True,
    )
    parser.add_argument("--size", choices=["7b", "8b"], required=True)
    parser.add_argument("--group", choices=["AD", "HC"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiments.json"))
    parser.add_argument("--report-to", nargs="+", default=["none"])
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser.parse_args()


def build_text_dataset(frame, tokenizer):
    records = [
        {
            "text": format_chat_prompt(
                tokenizer=tokenizer,
                question=row["question"],
                answer=row["answer"],
                add_generation_prompt=False,
            )
        }
        for _, row in frame.iterrows()
    ]
    return Dataset.from_pandas(pd.DataFrame(records), preserve_index=False)


def load_chat_model(model_id, hf_token, load_in_4bit):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.config.use_cache = False
    return model, tokenizer


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    family_config, model_id = resolve_family_config(config, args.family, args.size)
    chat_cfg = config["chat_lora"]
    hf_token = os.environ.get(args.hf_token_env)

    frames = load_split_frames(args.data_dir, args.group)
    run_name = build_run_name(args.family, args.size, args.group)
    run_paths = get_run_paths(REPO_ROOT, run_name)
    run_paths["model_dir"].mkdir(parents=True, exist_ok=True)
    run_paths["log_dir"].mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_chat_model(model_id, hf_token, chat_cfg["load_in_4bit"])
    train_dataset = build_text_dataset(frames["train"], tokenizer)
    eval_dataset = build_text_dataset(frames["validation"], tokenizer)

    peft_config = LoraConfig(
        r=chat_cfg["lora_r"],
        lora_alpha=chat_cfg["lora_alpha"],
        lora_dropout=chat_cfg["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=chat_cfg["target_modules"],
    )

    training_args = TrainingArguments(
        output_dir=str(run_paths["model_dir"]),
        logging_dir=str(run_paths["log_dir"]),
        overwrite_output_dir=True,
        report_to=args.report_to,
        learning_rate=chat_cfg["learning_rate"],
        num_train_epochs=chat_cfg["num_train_epochs"],
        per_device_train_batch_size=chat_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=chat_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=chat_cfg["gradient_accumulation_steps"],
        weight_decay=chat_cfg["weight_decay"],
        save_total_limit=chat_cfg["save_total_limit"],
        evaluation_strategy=chat_cfg["evaluation_strategy"],
        save_strategy=chat_cfg["save_strategy"],
        logging_strategy=chat_cfg["logging_strategy"],
        optim=chat_cfg["optim"],
        lr_scheduler_type=chat_cfg["lr_scheduler_type"],
        warmup_ratio=chat_cfg["warmup_ratio"],
        max_grad_norm=chat_cfg["max_grad_norm"],
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_chat_length"],
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=chat_cfg["early_stopping_patience"])],
    )
    trainer.train()
    trainer.model.save_pretrained(str(run_paths["model_dir"]))
    tokenizer.save_pretrained(str(run_paths["model_dir"]))


if __name__ == "__main__":
    main()
