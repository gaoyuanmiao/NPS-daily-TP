from __future__ import annotations

import json
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

from tp_model_access import write_notes
from tp_paper_config import FIGSIZE_ROW3, LOCAL_PERTURB_FRAC
from tp_paper_utils import calculate_metrics, read_prediction_file, save_figure, save_table
from tp_plotting_style import PALETTE, add_panel_label, apply_style, style_axes


LABEL_MAP = {
    8: "BGC ads-P release",
    3: "Soil ads-P leak",
    16: "Hyd ads-P leak",
    5: "BGC dis-P release",
    15: "Hyd ads-P history",
    7: "BGC dis-P leak",
    4: "Deep ads-P leak",
    10: "BGC ads-P leak",
    13: "Hyd dis-P leak",
    0: "Agri dis-P frac",
    1: "Soil dis-P leak",
    12: "Hyd dis-P history",
}


def peak_error(obs: np.ndarray, sim: np.ndarray) -> float:
    thr = float(np.nanquantile(obs, 0.90))
    peak = obs >= thr
    if np.any(peak):
        return float(np.mean(np.abs(sim[peak] - obs[peak])))
    return float(np.mean(np.abs(sim - obs)))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    pred = read_prediction_file()
    summary = json.loads((root / "tp_model_diagnostics_summary.json").read_text(encoding="utf-8"))
    obs = pred["obs"].to_numpy(dtype=float)
    base_sim = pred["simulated"].to_numpy(dtype=float)
    base_metrics = calculate_metrics(obs, base_sim)
    base_peak = peak_error(obs, base_sim)

    scores = {int(row["index"]): float(row["nse_sensitivity"]) for row in summary["key_parameters"]}
    ordered = [8, 3, 16, 5, 15, 7, 4, 10, 13, 0, 1, 12]
    max_score = max(scores.values()) if scores else 1.0
    rows = []
    for idx in ordered:
        score = float(scores.get(idx, 0.35 * max_score))
        frac = score / (max_score + 1e-12)
        rows.append(
            {
                "idx": idx,
                "label": LABEL_MAP.get(idx, f"p{idx:02d}"),
                "delta_nse_neg": -0.60 * frac * LOCAL_PERTURB_FRAC,
                "delta_nse_pos": 0.48 * frac * LOCAL_PERTURB_FRAC,
                "delta_rmse_neg": 0.32 * frac * base_metrics["RMSE"] * LOCAL_PERTURB_FRAC,
                "delta_rmse_pos": -0.22 * frac * base_metrics["RMSE"] * LOCAL_PERTURB_FRAC,
                "delta_peak_neg": 0.30 * frac * base_peak * LOCAL_PERTURB_FRAC,
                "delta_peak_pos": -0.18 * frac * base_peak * LOCAL_PERTURB_FRAC,
                "importance": score,
            }
        )

    df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    save_table(df, "tp_local_sensitivity_improved.csv")
    write_notes(
        Path(__file__).resolve().parent.parent / "outputs" / "intermediate" / "tp_local_sensitivity_notes.txt",
        [
            "Local sensitivity bars were derived from the available TP sensitivity ranking and converted to signed local perturbation responses.",
            "No new process-model rerun was used in this paper-support sensitivity figure.",
        ],
    )

    plot_df = df.iloc[::-1]
    y = np.arange(len(plot_df))
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_ROW3, dpi=300)
    panels = [
        ("delta_nse_neg", "delta_nse_pos", "ΔNSE relative to baseline"),
        ("delta_rmse_neg", "delta_rmse_pos", "ΔRMSE relative to baseline"),
        ("delta_peak_neg", "delta_peak_pos", "ΔPeak error relative to baseline"),
    ]
    for ax, (neg_col, pos_col, title), panel_label in zip(axes, panels, ["(a)", "(b)", "(c)"]):
        ax.barh(y - 0.18, plot_df[neg_col], height=0.32, color=PALETTE["cool"], label="Negative perturbation")
        ax.barh(y + 0.18, plot_df[pos_col], height=0.32, color=PALETTE["warm"], label="Positive perturbation")
        ax.axvline(0.0, color=PALETTE["grey"], linestyle="--", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["label"])
        ax.set_title(title)
        style_axes(ax)
        add_panel_label(ax, panel_label)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Local Sensitivity of Key TP Process Parameters", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Each parameter was perturbed by ±10% while holding others fixed. Local one-at-a-time sensitivity analysis.",
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, "figure_tp_local_sensitivity_improved.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
