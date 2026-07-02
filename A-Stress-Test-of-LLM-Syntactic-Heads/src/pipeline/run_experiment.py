"""Runs one full experiment end to end: load model -> extract surprisal +
attention for every garden-path sentence -> apply Fix 1/2/3 -> compute H1
correlation and H2 crash/recovery -> dump results to results/<name>/.

Usage:
    python -m src.pipeline.run_experiment --config configs/gpt2_en.yaml
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.analysis.arc_coverage import arc_attention_weight
from src.analysis.head_identification import identify_syntactic_heads, max_confidence_over_heads
from src.analysis.recovery import crash_and_recovery, recovery_analysis
from src.analysis.residualization import surprisal_confidence_correlation
from src.analysis.surprisal import compute_surprisal


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg["model_family"] == "llama":
        from src.training.load_llama import load_llama
        model, tokenizer = load_llama()
        return model, tokenizer, device, cfg["n_layers"], cfg["n_heads"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], output_attentions=True).to(device)
    model.eval()
    return model, tokenizer, device, cfg["n_layers"], cfg["n_heads"]


def run(cfg):
    model, tokenizer, device, n_layers, n_heads = load_model(cfg)
    sentences = load_dataset(cfg["dataset_path"])

    # Pass 1: collect confidence matrices per token to identify syntactic heads (Fix 2)
    all_conf_matrices = []
    per_sentence_cache = []

    for sent in sentences:
        surp, attn = compute_surprisal(sent["text"], tokenizer, model, device)
        per_sentence_cache.append((sent, surp, attn))

        for tok in sent["deps"]:
            if tok["head"] == 0:
                continue
            mat = np.zeros((n_layers, n_heads))
            for l in range(n_layers):
                for h in range(n_heads):
                    mat[l, h] = arc_attention_weight(attn[l][0, h], tok["id"], tok["head"])
            all_conf_matrices.append(mat)

    syntactic_heads, acc_matrix, threshold = identify_syntactic_heads(all_conf_matrices, n_layers, n_heads)
    print(f"identified {len(syntactic_heads)}/{n_layers * n_heads} syntactic heads (threshold={threshold:.4f})")

    # Pass 2: build the per-token dataframe using only the identified syntactic heads
    rows = []
    for sent, surp, attn in per_sentence_cache:
        disambig_id = sent.get("disambig_token_id")
        by_id = {t["id"]: t for t in sent["deps"]}

        for tok in sent["deps"]:
            if tok["head"] == 0:
                continue
            conf = max_confidence_over_heads(attn, syntactic_heads, tok["id"], tok["head"], arc_attention_weight)
            rows.append({
                "sent_id": sent.get("sent_id"),
                "token_id": tok["id"],
                "deprel": tok["deprel"],
                "is_root": int(tok["deprel"] == "root"),
                "surprisal": surp.get(tok["id"], np.nan),
                "conf_max": conf,
                "offset": (tok["id"] - disambig_id) if disambig_id is not None else None,
            })

    df = pd.DataFrame(rows).dropna(subset=["surprisal", "conf_max"])

    # H1: surprisal-confidence correlation
    h1 = surprisal_confidence_correlation(df)

    # H2: crash and recovery around disambiguation
    conf_by_offset = recovery_analysis(df.to_dict("records"))
    h2 = crash_and_recovery(conf_by_offset)

    # H4 bonus: confidence by dependency relation (Figure 4)
    by_deprel = df.groupby("deprel")["conf_max"].mean().sort_values(ascending=False).to_dict()

    results = {
        "experiment": cfg["name"],
        "n_syntactic_heads": len(syntactic_heads),
        "n_total_heads": n_layers * n_heads,
        "threshold": threshold,
        "h1_correlation": {k: v for k, v in h1.items() if k not in ("surp_residual", "conf_residual")},
        "h2_crash_recovery": h2,
        "confidence_by_deprel": by_deprel,
        "syntactic_heads": syntactic_heads,
    }

    out_dir = os.path.join("results", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    df.to_csv(os.path.join(out_dir, "per_token.csv"), index=False)
    np.save(os.path.join(out_dir, "head_accuracy_matrix.npy"), acc_matrix)

    print(json.dumps(results["h1_correlation"], indent=2))
    print(json.dumps(results["h2_crash_recovery"], indent=2))
    print(f"saved full results to {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)


if __name__ == "__main__":
    main()
