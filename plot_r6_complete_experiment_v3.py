#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R6 Complete Experiment Analysis - v3
====================================

UPDATED FOR THE FINAL R6 BENCHMARK DIRECTORY:

    results_r6_complete/
    ├── Greedy/
    ├── Least-Loaded/
    ├── MAPPO-GNN/
    ├── MAPPO-No-GNN/
    ├── Random/
    ├── SA-PPO/
    ├── benchmark_seed_results.csv
    ├── benchmark_method_summary.csv
    ├── ablation_paired_statistics.csv
    └── config.json

The script is designed for the final six-method benchmark:

    MAPPO-GNN
    MAPPO-No-GNN
    SA-PPO
    Greedy
    Least-Loaded
    Random

RL methods with training/validation traces:
    MAPPO-GNN, MAPPO-No-GNN, SA-PPO

Final test:
    600 requests/seed
    seeds = 11, 22, 33, 44, 55

The script:
  1. discovers the six-method R6 directory automatically;
  2. loads training rewards where available;
  3. loads validation JSON files where available;
  4. loads final test data from benchmark_seed_results.csv when possible,
     otherwise falls back to per-seed test summary JSON files;
  5. generates publication-oriented figures;
  6. recomputes mean +/- sample SD from seed-level results;
  7. performs the controlled MAPPO-GNN vs MAPPO-No-GNN paired analysis;
  8. exports CSV and LaTeX tables.

IMPORTANT:
  - No fake validation reward is created.
  - Heuristic methods are NOT treated as trained RL models.
  - NaN policy-diff/entropy values for heuristic methods are preserved.
  - No colors are hard-coded; matplotlib's default style is used.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHODS = [
    "MAPPO-GNN",
    "MAPPO-No-GNN",
    "SA-PPO",
    "Greedy",
    "Least-Loaded",
    "Random",
]

RL_METHODS = ["MAPPO-GNN", "MAPPO-No-GNN", "SA-PPO"]
SEEDS = [11, 22, 33, 44, 55]
SMOOTHING_WINDOW = 50

METRICS = {
    "accepted_request_rate": ("Acceptance rate", "%", 100.0, "higher"),
    "mean_constraint_violation": ("Constraint violation", "", 1.0, "lower"),
    "mean_energy_J": ("Mean energy", "J", 1.0, "lower"),
    "mean_latency_ms": ("Mean latency", "ms", 1.0, "lower"),
    "p95_latency_ms": ("P95 latency", "ms", 1.0, "lower"),
    "mean_unique_nodes": ("Unique nodes", "", 1.0, "context"),
}

VALIDATION_METRICS = {
    "accepted_request_rate": ("Acceptance rate", "%", 100.0),
    "mean_constraint_violation": ("Mean constraint violation", "", 1.0),
    "mean_energy_J": ("Mean energy (J)", "J", 1.0),
    "mean_latency_ms": ("Mean latency (ms)", "ms", 1.0),
    "p95_latency_ms": ("P95 latency (ms)", "ms", 1.0),
    "mean_unique_nodes": ("Mean unique nodes", "", 1.0),
    "mean_pairwise_policy_diff": ("Mean pairwise policy difference", "", 1.0),
    "mean_agent_entropy": ("Mean agent entropy", "", 1.0),
}


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures and tables for the R6 six-method benchmark."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=here / "results_r6_complete",
        help="R6 results directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "figures_r6_complete_v3",
        help="Output directory for figures/tables.",
    )
    return parser.parse_args()


def die(msg: str):
    print("\nERROR:\n" + msg)
    sys.exit(1)


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {pdf.name}")
    print(f"  [OK] {png.name}")


def checkpoint_number(path: Path):
    m = re.search(r"validation_ep(\d+)", path.stem.lower())
    return int(m.group(1)) if m else None


def canonical_method(value):
    s = str(value).strip()
    key = re.sub(r"[\s_]+", "-", s).lower()
    aliases = {
        "mappo-gnn": "MAPPO-GNN",
        "mappo-nognn": "MAPPO-No-GNN",
        "mappo-no-gnn": "MAPPO-No-GNN",
        "sa-ppo": "SA-PPO",
        "greedy": "Greedy",
        "least-loaded": "Least-Loaded",
        "leastloaded": "Least-Loaded",
        "random": "Random",
    }
    return aliases.get(key, s)


def discover(results_root: Path):
    if not results_root.exists():
        die(
            f"Cannot find:\n  {results_root}\n\n"
            "Place this script beside results_r6_complete or use --results-root."
        )

    found = {}
    print("=" * 82)
    print("R6 FINAL BENCHMARK DISCOVERY")
    print("=" * 82)
    print("RESULTS ROOT:", results_root)

    for method in METHODS:
        method_dir = results_root / method
        found[method] = {}
        if not method_dir.exists():
            print(f"[WARNING] Missing method directory: {method_dir}")
            continue

        for seed in SEEDS:
            seed_dir = method_dir / f"seed{seed}"
            if seed_dir.exists():
                found[method][seed] = seed_dir
                print(f"[OK] {method:14s} seed{seed}: {seed_dir}")
            else:
                print(f"[WARNING] Missing: {seed_dir}")

    return found


def load_training(found):
    rows = []

    for method in RL_METHODS:
        for seed in SEEDS:
            d = found.get(method, {}).get(seed)
            if d is None:
                continue

            path = d / "training_rewards.csv"
            if not path.exists():
                print(f"[WARNING] Missing training rewards: {path}")
                continue

            df = pd.read_csv(path)

            reward_col = next(
                (
                    c
                    for c in [
                        "reward",
                        "episode_reward",
                        "total_reward",
                        "mean_reward",
                    ]
                    if c in df.columns
                ),
                None,
            )
            if reward_col is None:
                print(f"[WARNING] No reward column: {path}")
                continue

            episode_col = next(
                (
                    c
                    for c in ["episode", "episodes", "ep", "iteration"]
                    if c in df.columns
                ),
                None,
            )

            episode = (
                np.arange(1, len(df) + 1)
                if episode_col is None
                else pd.to_numeric(df[episode_col], errors="coerce")
            )
            reward = pd.to_numeric(df[reward_col], errors="coerce")

            x = pd.DataFrame(
                {
                    "method": method,
                    "seed": seed,
                    "episode": episode,
                    "reward": reward,
                    "source": str(path),
                }
            ).dropna(subset=["episode", "reward"])

            x["episode"] = x["episode"].astype(int)
            rows.append(x)

            print(
                f"[TRAIN] {method:14s} seed{seed}: {len(x)} episodes"
            )

    if not rows:
        return pd.DataFrame(
            columns=["method", "seed", "episode", "reward", "source"]
        )

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["method", "seed", "episode"])
        .reset_index(drop=True)
    )


def load_validation(found):
    rows = []

    fields = [
        "joint_feasibility",
        "accepted_request_rate",
        "accepted_requests_total",
        "rejected_requests_total",
        "joint_feasible_requests_total",
        "feasible_but_rejected_total",
        "infeasible_but_accepted_total",
        "feasibility_acceptance_gap",
        "mean_constraint_violation",
        "mean_energy_J",
        "mean_latency_ms",
        "p95_latency_ms",
        "mean_unique_nodes",
        "mean_pairwise_policy_diff",
        "mean_agent_entropy",
        "mask_relaxed_decisions",
    ]

    for method in RL_METHODS:
        for seed in SEEDS:
            d = found.get(method, {}).get(seed)
            if d is None:
                continue

            files = sorted(
                d.glob("validation_ep*.json"),
                key=lambda p: checkpoint_number(p) or 0,
            )

            print(
                f"[VAL]   {method:14s} seed{seed}: "
                f"{len(files)} checkpoints"
            )

            for path in files:
                ep = checkpoint_number(path)
                if ep is None:
                    continue

                try:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"[WARNING] {path}: {exc}")
                    continue

                row = {
                    "method": method,
                    "seed": seed,
                    "checkpoint": ep,
                    "source": str(path),
                }

                for field in fields:
                    value = obj.get(field, np.nan)
                    if isinstance(value, (int, float)) and not isinstance(
                        value, bool
                    ):
                        row[field] = float(value)
                    else:
                        row[field] = np.nan

                rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["method", "seed", "checkpoint"])
        .reset_index(drop=True)
    )


def load_final_test_from_benchmark(results_root: Path):
    """
    Preferred source:
        benchmark_seed_results.csv

    The loader is intentionally schema-tolerant because column names may
    differ slightly between experiment revisions.
    """
    path = results_root / "benchmark_seed_results.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize method column.
    method_col = next(
        (c for c in ["method", "Method", "METHOD"] if c in df.columns), None
    )
    if method_col is None:
        return pd.DataFrame()

    df["method"] = df[method_col].map(canonical_method)

    seed_col = next(
        (c for c in ["source_seed", "seed", "Seed"] if c in df.columns),
        None,
    )
    if seed_col is not None:
        df["seed"] = pd.to_numeric(df[seed_col], errors="coerce")
    else:
        df["seed"] = np.nan

    aliases = {
        "requests": ["requests", "num_requests", "test_requests"],
        "accepted_request_rate": [
            "accepted_request_rate",
            "acceptance_rate",
            "acceptance",
        ],
        "mean_constraint_violation": [
            "mean_constraint_violation",
            "constraint_violation",
            "violation",
        ],
        "mean_energy_J": [
            "mean_energy_J",
            "energy_J",
            "mean_energy",
            "energy",
        ],
        "mean_latency_ms": [
            "mean_latency_ms",
            "latency_ms",
            "mean_latency",
            "latency",
        ],
        "p95_latency_ms": [
            "p95_latency_ms",
            "p95_latency",
            "p95",
        ],
        "mean_unique_nodes": [
            "mean_unique_nodes",
            "unique_nodes",
            "mean_unique",
        ],
        "mean_pairwise_policy_diff": [
            "mean_pairwise_policy_diff",
            "policy_diff",
        ],
        "mean_agent_entropy": [
            "mean_agent_entropy",
            "agent_entropy",
            "entropy",
        ],
        "mask_relaxed_decisions": [
            "mask_relaxed_decisions",
            "mask_relaxations",
        ],
    }

    out = pd.DataFrame(index=df.index)
    out["method"] = df["method"]
    out["seed"] = df["seed"]

    for target, candidates in aliases.items():
        source = next((c for c in candidates if c in df.columns), None)
        out[target] = (
            pd.to_numeric(df[source], errors="coerce")
            if source is not None
            else np.nan
        )

    return out[out.method.isin(METHODS)].reset_index(drop=True)


def load_final_test_from_json(found):
    rows = []

    fields = [
        "requests",
        "accepted_request_rate",
        "mean_constraint_violation",
        "mean_energy_J",
        "mean_latency_ms",
        "p95_latency_ms",
        "mean_unique_nodes",
        "mean_pairwise_policy_diff",
        "mean_agent_entropy",
        "mask_relaxed_decisions",
    ]

    for method in METHODS:
        for seed in SEEDS:
            d = found.get(method, {}).get(seed)
            if d is None:
                continue

            files = sorted(d.glob("test_seed*.summary.json"))
            if not files:
                continue

            # Usually exactly one. Use the newest lexicographically if several.
            path = files[-1]

            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[WARNING] {path}: {exc}")
                continue

            row = {"method": method, "seed": seed}
            for field in fields:
                row[field] = obj.get(field, np.nan)
            rows.append(row)

    return pd.DataFrame(rows)


def load_final_test(results_root, found):
    df = load_final_test_from_benchmark(results_root)

    if not df.empty:
        print(
            f"[TEST] Loaded {len(df)} seed-method rows from "
            f"{results_root / 'benchmark_seed_results.csv'}"
        )
        return df

    print("[TEST] benchmark_seed_results.csv unavailable; using per-seed JSON.")
    df = load_final_test_from_json(found)

    if not df.empty:
        print(f"[TEST] Loaded {len(df)} seed-method rows from JSON summaries.")

    return df


def compute_summary(test):
    if test.empty:
        return pd.DataFrame()

    rows = []

    for method in METHODS:
        d = test[test.method == method].copy()
        if d.empty:
            continue

        row = {"method": method, "n_seeds": len(d)}

        for metric in METRICS:
            col = metric
            x = pd.to_numeric(d[col], errors="coerce").dropna()
            row[f"{col}_mean"] = x.mean() if len(x) else np.nan
            row[f"{col}_std"] = x.std(ddof=1) if len(x) > 1 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def write_latex_table(summary, out_dir):
    if summary.empty:
        return

    rows = []
    display = {
        "accepted_request_rate": ("Acceptance", 100.0, ".2f"),
        "mean_constraint_violation": ("Violation", 1.0, ".5f"),
        "mean_energy_J": ("Energy (J)", 1.0, ".3f"),
        "mean_latency_ms": ("Latency (ms)", 1.0, ".3f"),
        "p95_latency_ms": ("P95 (ms)", 1.0, ".3f"),
        "mean_unique_nodes": ("Unique nodes", 1.0, ".3f"),
    }

    for _, r in summary.iterrows():
        cells = [r["method"]]
        for metric, (_, scale, fmt) in display.items():
            mean = r[f"{metric}_mean"] * scale
            std = r[f"{metric}_std"] * scale
            cells.append(f"{mean:{fmt}} $\\pm$ {std:{fmt}}")
        rows.append(" & ".join(cells) + r" \\")

    latex = r"""\begin{table}[t]
\centering
\caption{Final performance over five independent seeds and 600 test requests per seed. Values are mean $\pm$ sample standard deviation.}
\label{tab:final_benchmark}
\begin{tabular}{lrrrrrr}
\hline
Method & Acceptance (\%) & Violation & Energy (J) & Latency (ms) & P95 (ms) & Unique nodes \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""

    (out_dir / "table_final_benchmark.tex").write_text(
        latex, encoding="utf-8"
    )


def plot_training_raw(train, out_dir):
    for method in RL_METHODS:
        d = train[train.method == method]
        if d.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for seed in SEEDS:
            x = d[d.seed == seed]
            if not x.empty:
                ax.plot(
                    x.episode,
                    x.reward,
                    linewidth=1.0,
                    alpha=0.55,
                    label=f"Seed {seed}",
                )

        ax.set_xlabel("Training episode")
        ax.set_ylabel("Episode reward")
        ax.set_title(f"Training reward trajectories: {method}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, out_dir, f"fig_training_raw_{method.replace('-', '_').lower()}")


def plot_training_convergence(train, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    plotted = False
    for method in RL_METHODS:
        d = train[train.method == method]
        if d.empty:
            continue

        max_ep = int(d.groupby("seed").episode.max().min())
        grid = np.arange(1, max_ep + 1)
        series = []

        for seed in SEEDS:
            x = d[d.seed == seed]
            if x.empty:
                continue

            y = (
                x.set_index("episode")
                .reward.reindex(grid)
                .rolling(SMOOTHING_WINDOW, min_periods=10)
                .mean()
            )
            series.append(y.rename(seed))

        if not series:
            continue

        wide = pd.concat(series, axis=1)
        mean = wide.mean(axis=1)
        std = wide.std(axis=1)

        ax.plot(grid, mean, linewidth=2, label=method)
        ax.fill_between(
            grid, mean - std, mean + std, alpha=0.15
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Training episode")
    ax.set_ylabel("50-episode moving-average reward")
    ax.set_title("Cross-seed training convergence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_training_convergence_three_rl_methods")


def phase_summary(train):
    rows = []

    for method in RL_METHODS:
        for seed in SEEDS:
            d = train[
                (train.method == method) & (train.seed == seed)
            ].sort_values("episode")

            if d.empty:
                continue

            n = len(d)
            k = min(100, max(1, n // 2))
            first = d.head(k).reward
            last = d.tail(k).reward

            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "first100_mean": first.mean(),
                    "last100_mean": last.mean(),
                    "last100_std": last.std(),
                    "delta_reward": last.mean() - first.mean(),
                    "last50_mean": d.tail(min(50, n)).reward.mean(),
                    "last50_std": d.tail(min(50, n)).reward.std(),
                }
            )

    return pd.DataFrame(rows)


def plot_early_late(phase, out_dir):
    if phase.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.4))

    x = np.arange(len(RL_METHODS))
    width = 0.24

    for i, method in enumerate(RL_METHODS):
        d = phase[phase.method == method]
        first = d.first100_mean.mean()
        last = d.last100_mean.mean()

        ax.bar(
            i - width / 2,
            first,
            width=width,
            label="First 100" if i == 0 else None,
        )
        ax.bar(
            i + width / 2,
            last,
            width=width,
            label="Last 100" if i == 0 else None,
        )

    ax.set_xticks(x, RL_METHODS, rotation=10)
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean reward")
    ax.set_title("Early-to-late training reward")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_training_early_vs_late_rl")


def plot_training_improvement(phase, out_dir):
    if phase.empty:
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(SEEDS))
    width = 0.24

    for i, method in enumerate(RL_METHODS):
        d = phase[phase.method == method].set_index("seed")
        vals = [d.loc[s, "delta_reward"] if s in d.index else np.nan for s in SEEDS]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=method,
        )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x, [str(s) for s in SEEDS])
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Δ reward (last 100 − first 100)")
    ax.set_title("Training improvement across seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_training_reward_improvement")


def plot_training_stability(phase, out_dir):
    if phase.empty:
        return

    d = phase.copy()
    d["cv"] = d.last100_std / d.last100_mean.abs()

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(RL_METHODS))
    means = []
    stds = []

    for method in RL_METHODS:
        vals = d.loc[d.method == method, "cv"].dropna()
        means.append(vals.mean() if len(vals) else np.nan)
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x, RL_METHODS, rotation=10)
    ax.set_xlabel("Method")
    ax.set_ylabel("Late-training coefficient of variation")
    ax.set_title("Late-training reward stability")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_training_stability_rl")


def plot_validation(val, column, ylabel, title, filename, out_dir):
    if val.empty or column not in val.columns:
        return
    if val[column].notna().sum() == 0:
        print(f"[WARNING] Validation metric unavailable: {column}")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for method in RL_METHODS:
        d = val[val.method == method]
        if d.empty:
            continue

        # First average over seeds at each checkpoint.
        g = (
            d.groupby("checkpoint")[column]
            .agg(["mean", "std"])
            .reset_index()
        )

        ax.plot(
            g.checkpoint,
            g["mean"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=method,
        )
        ax.fill_between(
            g.checkpoint,
            g["mean"] - g["std"],
            g["mean"] + g["std"],
            alpha=0.15,
        )

    ax.set_xlabel("Training checkpoint (episode)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, filename)


def plot_all_validation(val, out_dir):
    for col, (ylabel, _, _) in VALIDATION_METRICS.items():
        plot_validation(
            val,
            col,
            ylabel,
            f"Validation {ylabel.lower()} across checkpoints",
            f"fig_validation_{col}",
            out_dir,
        )


def plot_final_metric(test, metric, out_dir):
    if test.empty or metric not in test.columns:
        return

    title, unit, scale, direction = METRICS[metric]

    means = []
    stds = []

    for method in METHODS:
        x = pd.to_numeric(
            test.loc[test.method == method, metric],
            errors="coerce",
        ).dropna() * scale

        means.append(x.mean() if len(x) else np.nan)
        stds.append(x.std(ddof=1) if len(x) > 1 else np.nan)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    pos = np.arange(len(METHODS))

    ax.bar(pos, means, yerr=stds, capsize=4)
    ax.set_xticks(pos, METHODS, rotation=15)
    ax.set_xlabel("Method")
    ax.set_ylabel(f"{title}{f' ({unit})' if unit else ''}")
    ax.set_title(f"Final 600-request evaluation: {title}")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save_fig(
        fig,
        out_dir,
        f"fig_final_{metric}",
    )


def plot_final_all(test, out_dir):
    for metric in METRICS:
        plot_final_metric(test, metric, out_dir)


def plot_tradeoff_energy_latency(test, out_dir):
    if test.empty:
        return

    rows = []
    for method in METHODS:
        d = test[test.method == method]
        if d.empty:
            continue

        rows.append(
            {
                "method": method,
                "energy": pd.to_numeric(d.mean_energy_J, errors="coerce").mean(),
                "latency": pd.to_numeric(d.mean_latency_ms, errors="coerce").mean(),
                "acceptance": 100
                * pd.to_numeric(
                    d.accepted_request_rate, errors="coerce"
                ).mean(),
            }
        )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ax.scatter(plot_df.energy, plot_df.latency, s=70)

    for _, r in plot_df.iterrows():
        ax.annotate(
            r["method"],
            (r["energy"], r["latency"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Mean energy (J)")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Energy-latency trade-off across six methods")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_final_energy_latency_tradeoff")


def plot_acceptance_violation_tradeoff(test, out_dir):
    if test.empty:
        return

    rows = []
    for method in METHODS:
        d = test[test.method == method]
        if d.empty:
            continue

        rows.append(
            {
                "method": method,
                "acceptance": 100
                * pd.to_numeric(
                    d.accepted_request_rate, errors="coerce"
                ).mean(),
                "violation": pd.to_numeric(
                    d.mean_constraint_violation, errors="coerce"
                ).mean(),
            }
        )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(plot_df.violation, plot_df.acceptance, s=70)

    for _, r in plot_df.iterrows():
        ax.annotate(
            r["method"],
            (r["violation"], r["acceptance"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Mean constraint violation")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_title("Acceptance-violation trade-off across six methods")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_final_acceptance_violation_tradeoff")


def paired_gnn_analysis(test, out_dir):
    """
    Controlled ablation:
        MAPPO-GNN vs MAPPO-No-GNN

    Uses the same five seeds and computes paired differences.
    If scipy is installed, also computes paired t-test and Wilcoxon.
    Holm correction is applied across the six metrics.
    """
    a = test[test.method == "MAPPO-GNN"].set_index("seed")
    b = test[test.method == "MAPPO-No-GNN"].set_index("seed")

    common = sorted(set(a.index).intersection(set(b.index)))
    if len(common) < 2:
        return pd.DataFrame()

    records = []

    for metric in METRICS:
        xa = pd.to_numeric(a.loc[common, metric], errors="coerce")
        xb = pd.to_numeric(b.loc[common, metric], errors="coerce")

        paired = pd.DataFrame({"a": xa, "b": xb}).dropna()
        if len(paired) < 2:
            continue

        diff = paired["a"] - paired["b"]

        rec = {
            "metric": metric,
            "n_pairs": len(diff),
            "gnn_mean": paired["a"].mean(),
            "nognn_mean": paired["b"].mean(),
            "mean_difference_gnn_minus_nognn": diff.mean(),
            "std_difference": diff.std(ddof=1),
            "raw_p_ttest": np.nan,
            "raw_p_wilcoxon": np.nan,
            "holm_p_ttest": np.nan,
            "holm_p_wilcoxon": np.nan,
        }

        try:
            from scipy.stats import ttest_rel, wilcoxon

            rec["raw_p_ttest"] = float(
                ttest_rel(paired["a"], paired["b"]).pvalue
            )

            try:
                rec["raw_p_wilcoxon"] = float(
                    wilcoxon(
                        paired["a"],
                        paired["b"],
                        zero_method="wilcox",
                        alternative="two-sided",
                    ).pvalue
                )
            except ValueError:
                rec["raw_p_wilcoxon"] = np.nan

        except ImportError:
            pass

        records.append(rec)

    result = pd.DataFrame(records)

    def holm(values):
        values = np.asarray(values, dtype=float)
        out = np.full(values.shape, np.nan)
        valid = np.where(np.isfinite(values))[0]
        if len(valid) == 0:
            return out

        order = valid[np.argsort(values[valid])]
        m = len(order)
        running = 0.0

        for rank, idx in enumerate(order):
            adjusted = min(1.0, (m - rank) * values[idx])
            running = max(running, adjusted)
            out[idx] = running

        return out

    result["holm_p_ttest"] = holm(result["raw_p_ttest"].values)
    result["holm_p_wilcoxon"] = holm(result["raw_p_wilcoxon"].values)

    result.to_csv(
        out_dir / "gnn_vs_nognn_paired_statistics_recomputed.csv",
        index=False,
    )

    return result


def plot_gnn_ablation_differences(paired, out_dir):
    if paired is None or paired.empty:
        return

    labels = [
        "Acceptance",
        "Violation",
        "Energy",
        "Latency",
        "P95 latency",
        "Unique nodes",
    ]

    # Use percentage points for acceptance to make the first effect readable.
    display_values = []
    display_labels = []

    for _, r in paired.iterrows():
        metric = r["metric"]
        value = r["mean_difference_gnn_minus_nognn"]

        if metric == "accepted_request_rate":
            value *= 100.0
        display_values.append(value)
        display_labels.append(
            {
                "accepted_request_rate": "Acceptance (pp)",
                "mean_constraint_violation": "Violation",
                "mean_energy_J": "Energy (J)",
                "mean_latency_ms": "Latency (ms)",
                "p95_latency_ms": "P95 latency (ms)",
                "mean_unique_nodes": "Unique nodes",
            }.get(metric, metric)
        )

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.axhline(0, linewidth=1)
    ax.bar(np.arange(len(display_values)), display_values)
    ax.set_xticks(
        np.arange(len(display_values)),
        display_labels,
        rotation=15,
    )
    ax.set_ylabel("GNN − No-GNN paired mean difference")
    ax.set_title("Controlled MAPPO-GNN ablation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_gnn_ablation_paired_differences")


def integrity_report(results_root, val, test):
    print("\n" + "=" * 82)
    print("DATA INTEGRITY CHECKS")
    print("=" * 82)

    expected_methods = set(METHODS)
    actual_methods = set(test.method.unique()) if not test.empty else set()

    missing_methods = expected_methods - actual_methods
    if not missing_methods:
        print("[OK] All six benchmark methods are present.")
    else:
        print("[WARNING] Missing final-test methods:", sorted(missing_methods))

    if not test.empty:
        if len(test) >= 30:
            print(f"[OK] Final seed-method rows: {len(test)}")
        else:
            print(f"[WARNING] Expected 30 rows, found {len(test)}")

        req = pd.to_numeric(test.requests, errors="coerce")
        if req.notna().all() and (req == 600).all():
            print("[OK] All final tests contain 600 requests.")
        else:
            print("[WARNING] Some final tests are not 600 requests.")

        relaxed = pd.to_numeric(
            test.mask_relaxed_decisions, errors="coerce"
        ).fillna(0)

        if (relaxed == 0).all():
            print("[OK] No mask-relaxed decisions in final tests.")
        else:
            print("[WARNING] Mask relaxation detected.")

    if not val.empty:
        expected = set(range(50, 801, 50))
        actual = set(val.checkpoint.dropna().astype(int).unique())
        if actual == expected:
            print("[OK] Validation checkpoints: 50..800 every 50.")
        else:
            print("[WARNING] Validation checkpoint set differs.")
            print("Expected:", sorted(expected))
            print("Actual  :", sorted(actual))

        gap = pd.to_numeric(
            val.feasibility_acceptance_gap, errors="coerce"
        ).fillna(0)

        if (gap.abs() <= 1e-12).all():
            print("[OK] Validation feasibility-acceptance gap = 0.")
        else:
            print("[WARNING] Non-zero validation feasibility-acceptance gap.")

    print("[INFO] Results root:", results_root)


def main():
    args = parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    found = discover(args.results_root)

    train = load_training(found)
    val = load_validation(found)
    test = load_final_test(args.results_root, found)

    if test.empty:
        die("No final benchmark results were loaded.")

    # Save normalized raw data.
    train.to_csv(args.out / "master_training_rewards.csv", index=False)
    if not val.empty:
        val.to_csv(args.out / "master_validation_results.csv", index=False)
    test.to_csv(args.out / "master_final_test_seed_results.csv", index=False)

    summary = compute_summary(test)
    summary.to_csv(
        args.out / "benchmark_method_summary_recomputed.csv",
        index=False,
    )
    write_latex_table(summary, args.out)

    print("\n" + "=" * 82)
    print("TRAINING FIGURES")
    print("=" * 82)
    if not train.empty:
        plot_training_raw(train, args.out)
        plot_training_convergence(train, args.out)
        phase = phase_summary(train)
        phase.to_csv(
            args.out / "training_phase_summary.csv",
            index=False,
        )
        plot_early_late(phase, args.out)
        plot_training_improvement(phase, args.out)
        plot_training_stability(phase, args.out)
    else:
        print("[WARNING] No RL training files found.")

    print("\n" + "=" * 82)
    print("VALIDATION FIGURES")
    print("=" * 82)
    if not val.empty:
        plot_all_validation(val, args.out)
    else:
        print("[WARNING] No validation JSON files found.")

    print("\n" + "=" * 82)
    print("FINAL BENCHMARK FIGURES")
    print("=" * 82)
    plot_final_all(test, args.out)
    plot_tradeoff_energy_latency(test, args.out)
    plot_acceptance_violation_tradeoff(test, args.out)

    print("\n" + "=" * 82)
    print("CONTROLLED GNN ABLATION")
    print("=" * 82)
    paired = paired_gnn_analysis(test, args.out)
    if not paired.empty:
        print(
            paired[
                [
                    "metric",
                    "mean_difference_gnn_minus_nognn",
                    "holm_p_ttest",
                    "holm_p_wilcoxon",
                ]
            ].round(6).to_string(index=False)
        )
        plot_gnn_ablation_differences(paired, args.out)

    integrity_report(args.results_root, val, test)

    print("\n" + "=" * 82)
    print("FINAL METHOD SUMMARY")
    print("=" * 82)

    display_cols = [
        "method",
        "n_seeds",
        "accepted_request_rate_mean",
        "accepted_request_rate_std",
        "mean_constraint_violation_mean",
        "mean_constraint_violation_std",
        "mean_energy_J_mean",
        "mean_energy_J_std",
        "mean_latency_ms_mean",
        "mean_latency_ms_std",
        "p95_latency_ms_mean",
        "p95_latency_ms_std",
        "mean_unique_nodes_mean",
        "mean_unique_nodes_std",
    ]

    print(summary[display_cols].round(5).to_string(index=False))

    print("\nOutput directory:")
    print(args.out.resolve())
    print("\nDone.")


if __name__ == "__main__":
    main()
