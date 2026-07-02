"""Regenerates Figures 1-5 from results/<experiment>/ output. Point this at
one experiment's results dir for the scatter + deprel plots, or at
results/ (the parent) for the cross-experiment summary (Figure 3).
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig1_scatter(results_dir, out_dir):
    df = pd.read_csv(os.path.join(results_dir, "per_token.csv"))
    with open(os.path.join(results_dir, "summary.json")) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["surprisal"], df["conf_max"], s=8, alpha=0.4)

    roots = df[df["is_root"] == 1]
    ax.scatter(roots["surprisal"], roots["conf_max"], s=20, marker="^", color="firebrick", label="Root")

    r = summary["h1_correlation"]["pearson_r"]
    ax.set_title(f"{summary['experiment']}  (r={r:.4f})")
    ax.set_xlabel("Surprisal (bits)")
    ax.set_ylabel("Attention confidence")
    ax.legend()

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{summary['experiment']}_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_recovery(results_dir, out_dir):
    with open(os.path.join(results_dir, "summary.json")) as f:
        summary = json.load(f)
    h2 = summary["h2_crash_recovery"]

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(["Pre-disambig", "At disambig", "Post-disambig"], [h2["pre"], h2["at"], h2["post"]],
                   color=["#4C72B0", "#C44E52", "#55A868"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                ha="center", va="bottom")
    ax.set_title(f"{summary['experiment']}  crash={h2['crash']:.3f}  recovery={h2['recovery']:.3f}\n{h2['pattern']}")
    ax.set_ylabel("Mean attention confidence")

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{summary['experiment']}_recovery.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_by_deprel(results_dir, out_dir):
    with open(os.path.join(results_dir, "summary.json")) as f:
        summary = json.load(f)
    by_deprel = summary["confidence_by_deprel"]

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(by_deprel.keys())
    vals = list(by_deprel.values())
    ax.barh(labels, vals, color="#4C72B0")
    ax.set_xlabel("Mean attention confidence")
    ax.set_title(f"{summary['experiment']} - confidence by dependency relation")
    ax.invert_yaxis()

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{summary['experiment']}_by_deprel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_cross_experiment_summary(results_root, out_dir):
    """Bar chart comparing Pearson r / Spearman rho across every experiment
    found under results_root."""
    rows = []
    for name in sorted(os.listdir(results_root)):
        summary_path = os.path.join(results_root, name, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            s = json.load(f)
        rows.append({
            "experiment": s["experiment"],
            "pearson_r": s["h1_correlation"]["pearson_r"],
            "spearman_rho": s["h1_correlation"]["spearman_rho"],
            "syn_heads": s["n_syntactic_heads"],
            "total_heads": s["n_total_heads"],
        })

    if not rows:
        print(f"no summary.json files found under {results_root}")
        return

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x - width / 2, df["pearson_r"], width, label="Pearson r")
    axes[0].bar(x + width / 2, df["spearman_rho"], width, label="Spearman rho", hatch="//")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["experiment"], rotation=20)
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_title("H1: Surprisal-Confidence Correlation")
    axes[0].legend()

    pct = 100 * df["syn_heads"] / df["total_heads"]
    axes[1].bar(x, pct, color="#DD8452")
    for i, v in enumerate(pct):
        axes[1].text(i, v, f"{df['syn_heads'][i]}/{df['total_heads'][i]}\n({v:.1f}%)", ha="center", va="bottom")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["experiment"], rotation=20)
    axes[1].set_title("H2: Syntactic Head Proportion")

    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cross_experiment_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results/<experiment> for single-experiment figures, or results/ for the cross-experiment summary")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.results, "summary.json")):
        fig1_scatter(args.results, args.out)
        fig2_recovery(args.results, args.out)
        fig4_by_deprel(args.results, args.out)
    else:
        fig3_cross_experiment_summary(args.results, args.out)

    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
