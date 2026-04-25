"""Visualize RSL-RL evaluation summaries and rank checkpoints.

Examples:

    python scripts/rsl_rl/plot_eval.py

    python scripts/rsl_rl/plot_eval.py \
        --summary_csv logs/rsl_rl/eval/eval_checkpoint_summary_20260425_162752.csv \
        --output_dir logs/rsl_rl/eval/plots

    python scripts/rsl_rl/plot_eval.py \
        --motion_keys body_pos body_rot joint_pos joint_vel \
        --success_weight 0.5 --return_weight 0.2 --motion_weight 0.3
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


DEFAULT_EVAL_DIR = Path("logs/rsl_rl/eval")
DEFAULT_MOTION_KEYS = ("body_pos", "body_rot", "joint_pos", "joint_vel")


parser = argparse.ArgumentParser(description="Plot evaluation summary CSVs and rank checkpoints.")
parser.add_argument(
    "--summary_csv",
    type=str,
    default=None,
    help="Path to eval_checkpoint_summary_*.csv. Defaults to the newest one under logs/rsl_rl/eval.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write plots and ranked_models.csv. Defaults to <summary_csv_dir>/plots_<timestamp>.",
)
parser.add_argument(
    "--motion_keys",
    type=str,
    nargs="*",
    default=list(DEFAULT_MOTION_KEYS),
    help="Motion metric suffixes to include in ranking, e.g. body_pos body_rot joint_pos joint_vel.",
)
parser.add_argument("--success_weight", type=float, default=0.5, help="Composite rank weight for success rate.")
parser.add_argument("--return_weight", type=float, default=0.2, help="Composite rank weight for return.")
parser.add_argument("--motion_weight", type=float, default=0.3, help="Composite rank weight for motion errors.")
parser.add_argument("--top_k", type=int, default=10, help="Number of top checkpoints to highlight in the text report.")


def _latest_summary_csv(eval_dir: Path) -> Path:
    candidates = sorted(eval_dir.glob("*checkpoint_summary*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint summary CSV found under {eval_dir}")
    return candidates[0]


def _to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _numeric_column(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [_to_float(row.get(key)) for row in rows]


def _finite(values: list[float]) -> list[float]:
    return [x for x in values if math.isfinite(x)]


def _normalize(values: list[float], higher_is_better: bool) -> list[float]:
    finite = _finite(values)
    if not finite:
        return [0.0 for _ in values]
    lo = min(finite)
    hi = max(finite)
    if math.isclose(lo, hi):
        return [1.0 if math.isfinite(x) else 0.0 for x in values]
    out = []
    for value in values:
        if not math.isfinite(value):
            out.append(0.0)
            continue
        score = (value - lo) / (hi - lo)
        out.append(score if higher_is_better else 1.0 - score)
    return out


def _find_motion_columns(rows: list[dict[str, Any]], motion_keys: list[str]) -> list[str]:
    headers = rows[0].keys()
    selected = []
    for key in motion_keys:
        column = f"motion_error_{key}_mean"
        if column in headers:
            selected.append(column)

    if selected:
        return selected

    return sorted(
        header
        for header in headers
        if header.startswith("motion_error_") and header.endswith("_mean")
    )


def _compute_scores(
    rows: list[dict[str, Any]],
    motion_columns: list[str],
    success_weight: float,
    return_weight: float,
    motion_weight: float,
) -> list[dict[str, Any]]:
    success_values = _numeric_column(rows, "success_rate_mean")
    return_values = _numeric_column(rows, "return_mean")

    success_scores = _normalize(success_values, higher_is_better=True)
    return_scores = _normalize(return_values, higher_is_better=True)

    motion_scores_per_col = []
    for column in motion_columns:
        motion_scores_per_col.append(_normalize(_numeric_column(rows, column), higher_is_better=False))

    ranked = []
    total_weight = success_weight + return_weight + (motion_weight if motion_scores_per_col else 0.0)
    if total_weight <= 0.0:
        total_weight = 1.0

    for idx, row in enumerate(rows):
        motion_score = 0.0
        if motion_scores_per_col:
            motion_score = sum(scores[idx] for scores in motion_scores_per_col) / len(motion_scores_per_col)

        composite = (
            success_weight * success_scores[idx]
            + return_weight * return_scores[idx]
            + (motion_weight * motion_score if motion_scores_per_col else 0.0)
        ) / total_weight

        ranked.append(
            {
                **row,
                "score": composite,
                "score_success_component": success_scores[idx],
                "score_return_component": return_scores[idx],
                "score_motion_component": motion_score,
            }
        )

    ranked.sort(key=lambda row: _to_float(row["score"]), reverse=True)
    return ranked


def _write_ranked_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _plot_success_return(rows: list[dict[str, Any]], out_path: Path) -> None:
    checkpoints = [row["checkpoint"] for row in rows]
    success = _numeric_column(rows, "success_rate_mean")
    success_ci = _numeric_column(rows, "success_rate_ci95")
    returns = _numeric_column(rows, "return_mean")

    fig, ax1 = plt.subplots(figsize=(max(8, len(rows) * 1.1), 5))
    x = list(range(len(rows)))
    ax1.bar(x, success, yerr=success_ci, capsize=4, color="#4C78A8", alpha=0.85, label="success rate")
    ax1.set_ylabel("success rate")
    ax1.set_ylim(0.0, max(1.0, max(_finite(success) or [1.0]) * 1.15))
    ax1.set_xticks(x)
    ax1.set_xticklabels(checkpoints, rotation=35, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    if _finite(returns):
        ax2 = ax1.twinx()
        ax2.plot(x, returns, color="#F58518", marker="o", linewidth=2.0, label="return")
        ax2.set_ylabel("return mean")

    ax1.set_title("Checkpoint Success Rate and Return")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_motion_errors(rows: list[dict[str, Any]], motion_columns: list[str], out_path: Path) -> None:
    if not motion_columns:
        return

    checkpoints = [row["checkpoint"] for row in rows]
    fig, axes = plt.subplots(
        len(motion_columns),
        1,
        figsize=(max(8, len(rows) * 1.1), max(3, len(motion_columns) * 2.2)),
        sharex=True,
    )
    if len(motion_columns) == 1:
        axes = [axes]

    x = list(range(len(rows)))
    for ax, column in zip(axes, motion_columns, strict=False):
        values = _numeric_column(rows, column)
        ci_column = column.removesuffix("_mean") + "_ci95"
        ci = _numeric_column(rows, ci_column) if ci_column in rows[0] else [0.0 for _ in rows]
        label = column.removeprefix("motion_error_").removesuffix("_mean")
        ax.errorbar(x, values, yerr=ci, color="#54A24B", marker="o", capsize=4)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(checkpoints, rotation=35, ha="right")
    fig.suptitle("Motion Tracking Errors (Lower Is Better)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ranked_scores(rows: list[dict[str, Any]], out_path: Path) -> None:
    checkpoints = [row["checkpoint"] for row in rows]
    scores = _numeric_column(rows, "score")

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.1), 5))
    x = list(range(len(rows)))
    ax.bar(x, scores, color="#B279A2", alpha=0.9)
    ax.set_ylabel("composite score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Model Selection Score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_report(path: Path, rows: list[dict[str, Any]], top_k: int, motion_columns: list[str]) -> None:
    lines = [
        "# Evaluation Model Ranking",
        "",
        f"Motion metrics used: {', '.join(motion_columns) if motion_columns else 'none'}",
        "",
        "| rank | checkpoint | score | success_rate | return | episode_length |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows[:top_k], start=1):
        lines.append(
            "| {rank} | {checkpoint} | {score:.4f} | {success:.4f} | {ret:.4f} | {length:.2f} |".format(
                rank=idx,
                checkpoint=row["checkpoint"],
                score=_to_float(row.get("score")),
                success=_to_float(row.get("success_rate_mean")),
                ret=_to_float(row.get("return_mean")),
                length=_to_float(row.get("episode_length_mean")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv) if args.summary_csv else _latest_summary_csv(DEFAULT_EVAL_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else summary_csv.parent / f"plots_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(summary_csv)
    motion_columns = _find_motion_columns(rows, args.motion_keys)
    ranked_rows = _compute_scores(
        rows,
        motion_columns,
        success_weight=args.success_weight,
        return_weight=args.return_weight,
        motion_weight=args.motion_weight,
    )

    ranked_csv = output_dir / "ranked_models.csv"
    _write_ranked_csv(ranked_csv, ranked_rows)

    _plot_success_return(rows, output_dir / "success_return.png")
    _plot_motion_errors(rows, motion_columns, output_dir / "motion_errors.png")
    _plot_ranked_scores(ranked_rows, output_dir / "model_selection_score.png")
    _write_report(output_dir / "report.md", ranked_rows, args.top_k, motion_columns)

    print(f"[INFO] Summary CSV: {os.path.abspath(summary_csv)}")
    print(f"[INFO] Output dir: {os.path.abspath(output_dir)}")
    print(f"[INFO] Ranked CSV: {os.path.abspath(ranked_csv)}")
    print(f"[INFO] Best checkpoint: {ranked_rows[0]['checkpoint']} (score={_to_float(ranked_rows[0]['score']):.4f})")


if __name__ == "__main__":
    main()
