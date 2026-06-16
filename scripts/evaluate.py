import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ad_narratives.shared import (  # noqa: E402
    build_run_name,
    ensure_nltk,
    get_run_paths,
    load_config,
    load_split_frames,
    resolve_family_config,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["flan-t5", "t5", "gpt2"], required=True)
    parser.add_argument("--size", choices=["small", "base", "large"], required=True)
    parser.add_argument("--group", choices=["AD", "HC"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiments.json"))
    parser.add_argument("--combine-splits", action="store_true")
    parser.add_argument("--model-dir")
    return parser.parse_args()


def load_model_and_tokenizer(model_dir, model_type):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_answer(model, tokenizer, device, model_type, prompt, generation_config, prefix):
    if model_type == "seq2seq":
        input_text = prefix + prompt
    else:
        input_text = f"Question: {prompt} Answer:"

    encoded = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    output = model.generate(
        **encoded,
        num_beams=generation_config["num_beams"],
        max_new_tokens=generation_config["max_new_tokens"],
        min_new_tokens=generation_config["min_new_tokens"],
        no_repeat_ngram_size=generation_config["no_repeat_ngram_size"],
        top_k=generation_config["top_k"],
        top_p=generation_config["top_p"],
        temperature=generation_config["temperature"],
        do_sample=generation_config["do_sample"],
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if model_type == "causal" and "Answer:" in decoded:
        decoded = decoded.split("Answer:", 1)[1].strip()
    return decoded


def compute_metrics(predictions, references):
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1
    metrics = {"rouge1": [], "rouge2": [], "rougeL": [], "bleu": []}
    for prediction, reference in zip(predictions, references):
        scores = rouge.score(reference, prediction)
        metrics["rouge1"].append(scores["rouge1"].fmeasure)
        metrics["rouge2"].append(scores["rouge2"].fmeasure)
        metrics["rougeL"].append(scores["rougeL"].fmeasure)
        metrics["bleu"].append(
            sentence_bleu(
                [reference.split()],
                prediction.split(),
                smoothing_function=smoother,
            )
        )
    return {name: float(sum(values) / len(values)) for name, values in metrics.items()}


def main():
    args = parse_args()
    ensure_nltk()
    config = load_config(args.config)
    set_seed(config["seed"])
    family_config, _ = resolve_family_config(config, args.family, args.size)
    run_name = build_run_name(args.family, args.size, args.group)
    run_paths = get_run_paths(REPO_ROOT, run_name)

    frames = load_split_frames(args.data_dir, args.group)
    if args.combine_splits:
        eval_frame = pd.concat([frames["train"], frames["validation"], frames["test"]], ignore_index=True)
    else:
        eval_frame = frames["test"]

    model_dir = Path(args.model_dir) if args.model_dir else run_paths["model_dir"]
    model, tokenizer, device = load_model_and_tokenizer(model_dir, family_config["type"])
    predictions = [
        generate_answer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_type=family_config["type"],
            prompt=question,
            generation_config=config["generation"],
            prefix=config["prefix"],
        )
        for question in eval_frame["question"]
    ]

    run_paths["prediction_dir"].mkdir(parents=True, exist_ok=True)
    run_paths["metric_dir"].mkdir(parents=True, exist_ok=True)
    prediction_path = run_paths["prediction_dir"] / f"{run_name}_predictions.csv"
    metric_path = run_paths["metric_dir"] / f"{run_name}_metrics.json"

    output_frame = eval_frame.copy()
    output_frame["generated_answer"] = predictions
    output_frame.to_csv(prediction_path, index=False)

    metrics = compute_metrics(predictions, output_frame["answer"].tolist())
    with open(metric_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps({"predictions": str(prediction_path), "metrics": str(metric_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()

