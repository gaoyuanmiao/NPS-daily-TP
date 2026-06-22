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

from tp_model_access import build_full_model_prediction_df, build_process_ensemble, write_notes
from tp_paper_config import FIGSIZE_ROW3, INTER_DIR, TP_PLOT_UNIT
from tp_paper_utils import calculate_metrics, save_figure, save_table
from tp_plotting_style import PALETTE, add_panel_label, apply_style, style_axes


METRICS_CSV = INTER_DIR / "tp_stability_member_metrics.csv"
NOTES_TXT = INTER_DIR / "tp_stability_notes.txt"


def main() -> None:
    base_df = build_full_model_prediction_df(force=True)
    ensemble_df = build_process_ensemble(force=True)
    member_cols = [c for c in ensemble_df.columns if c.startswith("member_")]
    obs = ensemble_df["obs"].to_numpy(dtype=float)

    rows = []
    for col in member_cols:
        sim = ensemble_df[col].to_numpy(dtype=float)
        metrics = calculate_metrics(obs, sim)
        rows.append({"member": col, **metrics})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            {
                "n_members": len(metrics_df),
                "nse_mean": float(metrics_df["NSE"].mean()),
                "nse_std": float(metrics_df["NSE"].std(ddof=0)),
                "r2_mean": float(metrics_df["R2"].mean()),
                "rmse_mean": float(metrics_df["RMSE"].mean()),
            }
        ]
    )
    save_table(summary, "tp_stability_statistics.csv")
    write_notes(
        NOTES_TXT,
        [
            "Stability metrics were computed from the process-guided TP proxy ensemble.",
            "A statistical proxy ensemble was used because no stored member-level TP ensemble was available.",
        ],
    )

    base_metrics = calculate_metrics(base_df["obs"], base_df["simulated"])
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_ROW3, dpi=300)
    ax1, ax2, ax3 = axes.ravel()

    sc = ax1.scatter(metrics_df["R2"], metrics_df["NSE"], c=np.abs(metrics_df["Bias"]), cmap="viridis", s=28, alpha=0.78, edgecolors="white", linewidths=0.25)
    ax1.scatter([base_metrics["R2"]], [base_metrics["NSE"]], marker="*", s=150, color="black", label="Baseline", zorder=4)
    ax1.axhline(base_metrics["NSE"], color=PALETTE["grey"], linestyle="--", linewidth=1.0)
    ax1.axvline(base_metrics["R2"], color=PALETTE["grey"], linestyle="--", linewidth=1.0)
    ax1.set_xlabel("R$^2$ (-)")
    ax1.set_ylabel("NSE (-)")
    ax1.set_title("Stability scatter of process-informed TP simulations")
    ax1.legend(frameon=False, loc="lower left")
    style_axes(ax1)
    add_panel_label(ax1, "(a)")
    cb = plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label(f"Absolute mean bias ({TP_PLOT_UNIT})")

    ax2.hist(metrics_df["NSE"], bins=18, color=PALETTE["cool"], alpha=0.78, edgecolor="white")
    ax2.axvline(base_metrics["NSE"], color="black", linewidth=1.4, label="Baseline")
    ax2.set_title("Distribution of NSE")
    ax2.set_xlabel("NSE (-)")
    ax2.set_ylabel("Frequency")
    ax2.legend(frameon=False)
    style_axes(ax2)
    add_panel_label(ax2, "(b)")

    ax3.hist(metrics_df["R2"], bins=18, color=PALETTE["warm"], alpha=0.78, edgecolor="white")
    ax3.axvline(base_metrics["R2"], color="black", linewidth=1.4, label="Baseline")
    ax3.set_title("Distribution of R$^2$")
    ax3.set_xlabel("R$^2$ (-)")
    ax3.set_ylabel("Frequency")
    ax3.legend(frameon=False)
    style_axes(ax3)
    add_panel_label(ax3, "(c)")

    fig.tight_layout()
    save_figure(fig, "figure_tp_stability_scatter_enhanced.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
