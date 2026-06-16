import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ad_narratives.shared import (  # noqa: E402
    build_run_name,
    get_run_paths,
    load_config,
    load_split_frames,
    resolve_family_config,
    set_seed,
)


class QADataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, frame, block_size):
        self.examples = []
        for _, row in frame.iterrows():
            text = f"Question: {row['question']} Answer: {row['answer']}"
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=block_size,
                padding="max_length",
            )
            self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {key: torch.tensor(value) for key, value in self.examples[index].items()}


def build_seq2seq_datasets(tokenizer, frames, prefix, max_source_length, max_target_length):
    def preprocess(examples):
        inputs = [prefix + question for question in examples["question"]]
        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)
        labels = tokenizer(examples["answer"], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    datasets = {}
    for split, frame in frames.items():
        dataset = Dataset.from_pandas(frame[["question", "answer"]], preserve_index=False)
        datasets[split] = dataset.map(preprocess, batched=True)
    return datasets


def train_seq2seq(args, config, model_id, run_paths):
    frames = load_split_frames(args.data_dir, args.group)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenized = build_seq2seq_datasets(
        tokenizer=tokenizer,
        frames=frames,
        prefix=config["prefix"],
        max_source_length=config["max_source_length"],
        max_target_length=config["max_target_length"],
    )

    train_cfg = config["training"].copy()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_paths["model_dir"]),
        logging_dir=str(run_paths["log_dir"]),
        overwrite_output_dir=True,
        report_to=args.report_to,
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        weight_decay=train_cfg["weight_decay"],
        save_total_limit=train_cfg["save_total_limit"],
        evaluation_strategy=train_cfg["evaluation_strategy"],
        save_strategy=train_cfg["save_strategy"],
        logging_strategy=train_cfg["logging_strategy"],
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_cfg["early_stopping_patience"])],
    )
    trainer.train()
    trainer.save_model(str(run_paths["model_dir"]))
    tokenizer.save_pretrained(str(run_paths["model_dir"]))


def train_causal(args, config, family_config, model_id, run_paths):
    frames = load_split_frames(args.data_dir, args.group)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)

    train_cfg = config["training"].copy()
    train_cfg.update(family_config.get("training", {}))
    train_dataset = QADataset(tokenizer, frames["train"], config["max_target_length"])
    val_dataset = QADataset(tokenizer, frames["validation"], config["max_target_length"])

    training_args = TrainingArguments(
        output_dir=str(run_paths["model_dir"]),
        logging_dir=str(run_paths["log_dir"]),
        overwrite_output_dir=True,
        report_to=args.report_to,
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        weight_decay=train_cfg["weight_decay"],
        save_total_limit=train_cfg["save_total_limit"],
        evaluation_strategy=train_cfg["evaluation_strategy"],
        save_strategy=train_cfg["save_strategy"],
        logging_strategy=train_cfg["logging_strategy"],
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_cfg["early_stopping_patience"])],
    )
    trainer.train()
    trainer.save_model(str(run_paths["model_dir"]))
    tokenizer.save_pretrained(str(run_paths["model_dir"]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["flan-t5", "t5", "gpt2"], required=True)
    parser.add_argument("--size", choices=["small", "base", "large"], required=True)
    parser.add_argument("--group", choices=["AD", "HC"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiments.json"))
    parser.add_argument("--report-to", nargs="+", default=["none"])
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    family_config, model_id = resolve_family_config(config, args.family, args.size)
    run_name = build_run_name(args.family, args.size, args.group)
    run_paths = get_run_paths(REPO_ROOT, run_name)
    for path in run_paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    if family_config["type"] == "seq2seq":
        train_seq2seq(args, config, model_id, run_paths)
    else:
        train_causal(args, config, family_config, model_id, run_paths)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

