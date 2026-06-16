import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_family_config(config, family, size):
    if family not in config["models"]:
        raise ValueError(f"Unknown family: {family}")
    family_config = config["models"][family]
    if size not in family_config["sizes"]:
        raise ValueError(f"Unknown size '{size}' for family '{family}'")
    return family_config, family_config["sizes"][size]


def build_run_name(family, size, group):
    return f"{family}_{size}_{group}"


def format_chat_prompt(tokenizer, question, answer=None, add_generation_prompt=False):
    messages = [{"role": "user", "content": str(question)}]
    if answer is not None:
        messages.append({"role": "assistant", "content": str(answer)})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    text = f"User: {question}"
    if answer is None:
        return text + "\nAssistant:"
    return text + f"\nAssistant: {answer}"


def get_data_paths(data_dir, group):
    data_dir = Path(data_dir)
    return {
        "train": data_dir / f"train_QA_{group}_data.csv",
        "validation": data_dir / f"val_QA_{group}_data.csv",
        "test": data_dir / f"test_QA_{group}_data.csv",
    }


def load_split_frames(data_dir, group):
    paths = get_data_paths(data_dir, group)
    frames = {}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} split: {path}")
        frame = pd.read_csv(path).astype(str)
        required = {"question", "answer"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        frames[split] = frame
    return frames


def get_run_paths(repo_root, run_name):
    repo_root = Path(repo_root)
    base = repo_root / "artifacts"
    return {
        "model_dir": base / "models" / run_name,
        "log_dir": base / "logs" / run_name,
        "prediction_dir": base / "predictions",
        "metric_dir": base / "metrics",
        "embedding_dir": base / "embeddings",
    }


def ensure_nltk():
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

