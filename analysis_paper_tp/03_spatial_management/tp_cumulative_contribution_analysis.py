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

from tp_paper_config import INTER_DIR
from tp_paper_utils import save_figure, save_table
from tp_plotting_style import PALETTE, add_panel_label, apply_style, style_axes


CONTRIB_MAP_NPY = INTER_DIR / "tp_annual_cell_contribution_map.npy"
CURVE_CSV = INTER_DIR / "tp_cumulative_contribution_curve.csv"
DIAG_TXT = INTER_DIR / "tp_cumulative_contribution_diagnostics.txt"


def main() -> None:
    contrib = np.load(CONTRIB_MAP_NPY)
    vals = contrib[np.isfinite(contrib) & (contrib > 0)]
    vals = np.sort(vals)[::-1]
    area_frac = np.arange(1, len(vals) + 1, dtype=float) / max(len(vals), 1)
    cum_frac = np.cumsum(vals) / (np.sum(vals) + 1e-12)
    curve = pd.DataFrame({"cumulative_area_fraction": area_frac, "cumulative_contribution_fraction": cum_frac})
    curve.to_csv(CURVE_CSV, index=False, encoding="utf-8-sig")

    def top_share(frac: float) -> float:
        n = max(1, int(np.ceil(frac * len(vals))))
        return float(np.sum(vals[:n]) / (np.sum(vals) + 1e-12))

    def area_needed(target: float) -> float:
        idx = int(np.searchsorted(cum_frac, target, side="left"))
        idx = min(idx, len(area_frac) - 1)
        return float(area_frac[idx])

    stats = pd.DataFrame(
        [
            {"metric": "top_1pct_contribution", "value": top_share(0.01)},
            {"metric": "top_5pct_contribution", "value": top_share(0.05)},
            {"metric": "top_10pct_contribution", "value": top_share(0.10)},
            {"metric": "top_20pct_contribution", "value": top_share(0.20)},
            {"metric": "area_for_50pct_contribution", "value": area_needed(0.50)},
            {"metric": "area_for_70pct_contribution", "value": area_needed(0.70)},
            {"metric": "area_for_80pct_contribution", "value": area_needed(0.80)},
            {"metric": "area_for_90pct_contribution", "value": area_needed(0.90)},
        ]
    )
    save_table(stats, "tp_cumulative_contribution_statistics.csv")
    DIAG_TXT.write_text(
        "\n".join(
            [
                f"Positive cells used: {len(vals)}",
                f"Annual contribution sum: {np.sum(vals):.6f}",
                f"Top 10% contribution share: {top_share(0.10):.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    apply_style()
    fig, ax = plt.subplots(figsize=(7.8, 6.2), dpi=300)
    ax.plot(area_frac, cum_frac, color=PALETTE["full_model"], linewidth=2.2, label="TP contribution curve")
    ax.plot([0, 1], [0, 1], color=PALETTE["grey"], linestyle="--", linewidth=1.0, label="1:1 reference")
    for frac, label in [(0.05, "Top 5%"), (0.10, "Top 10%"), (0.20, "Top 20%")]:
        share = top_share(frac)
        ax.scatter([frac], [share], color=PALETTE["warm"], s=34, zorder=3)
        ax.text(frac + 0.015, share - 0.035, label, fontsize=9, color="#334e68")
    ax.set_title("Spatial Concentration of TP Export Contributions")
    ax.set_xlabel("Cumulative area fraction")
    ax.set_ylabel("Cumulative TP export contribution")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    add_panel_label(ax, "(a)")
    fig.tight_layout()
    save_figure(fig, "figure_tp_cumulative_contribution_curve.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
