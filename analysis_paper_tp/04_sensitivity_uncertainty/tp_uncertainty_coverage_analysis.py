from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from tp_model_access import build_process_ensemble, write_notes
from tp_paper_config import ENSEMBLE_PREDICTIONS_CSV, FIGSIZE_GRID, INTER_DIR
from tp_paper_utils import identify_events, save_figure, save_table, split_train_val, value_label
from tp_plotting_style import PALETTE, add_panel_label, apply_style, style_axes


QUANT_CSV = INTER_DIR / "tp_uncertainty_quantiles.csv"
NOTES_TXT = INTER_DIR / "tp_uncertainty_notes.txt"


def compute_interval_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    member_cols = [c for c in df.columns if c.startswith("member_")]
    arr = df[member_cols].to_numpy(dtype=float)
    quant = pd.DataFrame(
        {
            "date": df["date"],
            "obs": df["obs"],
            "rain": df["rain"],
            "runoff": df["runoff"],
            "p05": np.percentile(arr, 5, axis=1),
            "p25": np.percentile(arr, 25, axis=1),
            "p50": np.percentile(arr, 50, axis=1),
            "p75": np.percentile(arr, 75, axis=1),
            "p95": np.percentile(arr, 95, axis=1),
        }
    )
    quant["width_50"] = quant["p75"] - quant["p25"]
    quant["width_90"] = quant["p95"] - quant["p05"]
    quant["cover_90"] = ((quant["obs"] >= quant["p05"]) & (quant["obs"] <= quant["p95"])).astype(int)
    quant = identify_events(quant)

    train_df, val_df = split_train_val(quant)
    groups = {
        "full_period": quant,
        "training_period": train_df,
        "validation_period": val_df,
        "peak_days": quant[quant["peak_TP_days"]],
        "normal_days": quant[~quant["peak_TP_days"]],
    }
    rows = []
    for name, subset in groups.items():
        if subset.empty:
            continue
        rows.append(
            {
                "group": name,
                "PICP_90": float(subset["cover_90"].mean()),
                "MPIW_90": float(subset["width_90"].mean()),
                "PICP_50": float(((subset["obs"] >= subset["p25"]) & (subset["obs"] <= subset["p75"])).mean()),
                "MPIW_50": float(subset["width_50"].mean()),
            }
        )
    stats = pd.DataFrame(rows)
    return quant, stats


def plot_uncertainty(quant: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.6), dpi=300)
    ax1, ax2 = axes.ravel()

    ax1.fill_between(quant["date"], quant["p05"], quant["p95"], color="#cfe8f3", alpha=0.95, label="90% interval")
    ax1.fill_between(quant["date"], quant["p25"], quant["p75"], color="#7fb3d5", alpha=0.70, label="50% interval")
    ax1.plot(quant["date"], quant["p50"], color=PALETTE["cool"], linewidth=1.9, label="Median")
    ax1.plot(quant["date"], quant["obs"], color=PALETTE["observed"], linewidth=1.4, label="Observed")
    ax1.set_title("Predictive intervals for daily TP")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(value_label())
    ax1.legend(frameon=False, ncol=4, loc="upper right")
    style_axes(ax1)
    add_panel_label(ax1, "(a)")

    ax2.plot(quant["date"], quant["cover_90"], color=PALETTE["cool"], linewidth=1.2, label="90% coverage")
    ax2.set_title("90% coverage indicator over time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Coverage indicator")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(frameon=False)
    style_axes(ax2)
    add_panel_label(ax2, "(b)")

    fig.tight_layout()
    save_figure(fig, "figure_tp_uncertainty_coverage_width.png")
    plt.close(fig)


def main() -> None:
    ensemble_df = build_process_ensemble(force=True)
    quant, stats = compute_interval_stats(ensemble_df)
    quant.to_csv(QUANT_CSV, index=False, encoding="utf-8-sig")
    save_table(stats, "tp_uncertainty_interval_statistics.csv")
    write_notes(
        NOTES_TXT,
        [
            f"Ensemble file: {ENSEMBLE_PREDICTIONS_CSV}",
            "Uncertainty intervals were derived from a process-guided TP proxy ensemble.",
            "A statistical proxy ensemble was used because no stored member-level TP ensemble was available.",
        ],
    )
    plot_uncertainty(quant)


if __name__ == "__main__":
    main()
