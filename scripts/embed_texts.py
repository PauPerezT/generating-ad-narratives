import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ad_narratives.shared import get_run_paths  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-name", default="google-bert/bert-base-uncased")
    parser.add_argument("--output")
    return parser.parse_args()


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def embed_texts(texts, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    vectors = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                str(text).lower(),
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            ).to(device)
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])[0].cpu().numpy()
            vectors.append(pooled)
    return vectors


def main():
    args = parse_args()
    input_path = Path(args.input)
    frame = pd.read_csv(input_path)
    if args.text_column not in frame.columns:
        raise ValueError(f"Missing text column '{args.text_column}' in {input_path}")

    embeddings = embed_texts(frame[args.text_column].astype(str).tolist(), args.model_name)
    embedding_frame = pd.DataFrame(embeddings)
    embedding_frame.insert(0, "label", args.label)
    embedding_frame.insert(0, "source_file", input_path.name)

    if args.output:
        output_path = Path(args.output)
    else:
        paths = get_run_paths(REPO_ROOT, args.label)
        paths["embedding_dir"].mkdir(parents=True, exist_ok=True)
        output_path = paths["embedding_dir"] / f"{input_path.stem}_{args.label}_embeddings.csv"

    embedding_frame.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
