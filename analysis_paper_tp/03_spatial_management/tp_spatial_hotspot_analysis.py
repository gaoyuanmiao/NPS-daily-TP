from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tp_model_access import build_full_model_spatial_data
from tp_paper_config import FIGSIZE_GRID, INTER_DIR, TP_SPATIAL_UNIT
from tp_paper_utils import save_figure, save_table
from tp_plotting_style import add_panel_label, apply_style, style_axes
from tp_daily_loader import load_tp_daily_data


SOURCE_MAP_NPY = INTER_DIR / "tp_annual_source_map.npy"
CONTRIB_MAP_NPY = INTER_DIR / "tp_annual_cell_contribution_map.npy"
ACC_MAP_NPY = INTER_DIR / "tp_annual_accumulated_flux_map.npy"
SUMMARY_JSON = INTER_DIR / "tp_spatial_hotspot_summary.json"


def _source_diagnostics(source_map: np.ndarray, landuse: np.ndarray) -> pd.DataFrame:
    rows = []
    for lu in sorted(int(v) for v in np.unique(landuse) if v > 0):
        mask = landuse == lu
        vals = source_map[mask]
        pos = vals[vals > 0]
        rows.append(
            {
                "landuse": lu,
                "cell_count": int(np.sum(mask)),
                "positive_source_cell_count": int(np.sum(vals > 0)),
                "source_sum": float(np.nansum(vals)),
                "source_mean_positive": float(np.nanmean(pos)) if pos.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _source_export_diagnostics(source_map: np.ndarray, contribution_map: np.ndarray, flux_map: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "annual_source_map_sum": float(np.nansum(source_map)),
                "annual_source_map_max": float(np.nanmax(source_map)),
                "annual_cell_contribution_sum": float(np.nansum(contribution_map)),
                "annual_cell_contribution_max": float(np.nanmax(contribution_map)),
                "annual_accumulated_flux_sum": float(np.nansum(flux_map)),
                "annual_accumulated_flux_max": float(np.nanmax(flux_map)),
            }
        ]
    )


def _build_local_output_proxy(
    source_map: np.ndarray,
    flow_acc: np.ndarray,
    dist_to_stream: np.ndarray,
    stream_mask: np.ndarray,
    landuse: np.ndarray,
    route_mask: np.ndarray,
    annual_total: float,
) -> np.ndarray:
    flow_norm = np.zeros_like(flow_acc, dtype=float)
    valid_flow = np.isfinite(flow_acc) & (flow_acc > 0)
    if np.any(valid_flow):
        flow_norm[valid_flow] = np.log1p(flow_acc[valid_flow]) / (np.nanmax(np.log1p(flow_acc[valid_flow])) + 1e-12)

    dist_norm = np.zeros_like(dist_to_stream, dtype=float)
    valid_dist = np.isfinite(dist_to_stream)
    dist_norm[valid_dist] = 1.0 / (1.0 + dist_to_stream[valid_dist])

    landuse_weight = np.ones_like(source_map, dtype=float) * 0.70
    landuse_weight[landuse == 1] = 1.00
    landuse_weight[landuse == 8] = 1.20
    landuse_weight[(landuse == 2) | (landuse == 4)] = 0.0

    export_factor = np.clip(
        (0.18 + 0.34 * flow_norm + 0.24 * dist_norm + 0.16 * np.asarray(stream_mask, dtype=float)) * landuse_weight,
        0.0,
        1.6,
    )
    local_output = np.where(route_mask, np.clip(source_map, 0.0, None) * export_factor, np.nan)
    local_output = np.where(route_mask & np.isfinite(local_output), local_output, np.nan)
    scale = annual_total / (np.nansum(local_output) + 1e-12)
    return np.where(route_mask, local_output * scale, np.nan)


def _build_acyclic_upstream(flowdir_up: dict[str, list[str]], outlet_code: str) -> tuple[list[str], dict[str, list[str]], int]:
    upstream: dict[str, list[str]] = {}
    topo: list[str] = []
    state: dict[str, int] = {}
    assigned_downstream: dict[str, str] = {}
    dropped_cycle_edges = 0

    def visit(node: str) -> None:
        nonlocal dropped_cycle_edges
        node_state = state.get(node, 0)
        if node_state == 2:
            return
        if node_state == 1:
            return
        state[node] = 1
        kept_ups: list[str] = []
        for up in flowdir_up.get(node, []):
            up = str(up)
            existing_downstream = assigned_downstream.get(up)
            if existing_downstream is not None and existing_downstream != node:
                continue
            assigned_downstream[up] = node
            up_state = state.get(up, 0)
            if up_state == 1:
                dropped_cycle_edges += 1
                continue
            visit(up)
            if state.get(up, 0) == 2:
                kept_ups.append(up)
        upstream[node] = kept_ups
        state[node] = 2
        topo.append(node)

    visit(str(outlet_code))
    return topo, upstream, dropped_cycle_edges


def _accumulate_routed_output(local_output_map: np.ndarray, flowdir_up: dict, study_mask: np.ndarray, outlet_code: str) -> tuple[np.ndarray, int]:
    topo, upstream_acyclic, dropped_cycle_edges = _build_acyclic_upstream(flowdir_up, str(outlet_code))
    memo: dict[str, float] = {}

    def solve(cell_id: str) -> float:
        if cell_id in memo:
            return memo[cell_id]
        i = int(cell_id[:4])
        j = int(cell_id[4:])
        if not (0 <= i < local_output_map.shape[0] and 0 <= j < local_output_map.shape[1] and study_mask[i, j]):
            memo[cell_id] = 0.0
            return 0.0
        val = local_output_map[i, j]
        total = float(val) if np.isfinite(val) else 0.0
        for up in upstream_acyclic.get(cell_id, []):
            total += solve(str(up))
        memo[cell_id] = total
        return total

    out = np.full_like(local_output_map, np.nan, dtype=float)
    for cell_id in topo:
        i = int(cell_id[:4])
        j = int(cell_id[4:])
        if 0 <= i < out.shape[0] and 0 <= j < out.shape[1] and study_mask[i, j]:
            out[i, j] = solve(str(cell_id))
    return out, dropped_cycle_edges


def _hotspot_masks(source_map: np.ndarray, contribution_map: np.ndarray, study_mask: np.ndarray):
    src_vals = source_map[np.isfinite(source_map) & (source_map > 0)]
    con_vals = contribution_map[np.isfinite(contribution_map) & (contribution_map > 0)]
    src_thr = float(np.nanpercentile(src_vals, 90)) if src_vals.size else np.nan
    con_thr = float(np.nanpercentile(con_vals, 90)) if con_vals.size else np.nan
    src_hot = np.isfinite(source_map) & (source_map > 0) & (source_map >= src_thr)
    con_hot = np.isfinite(contribution_map) & (contribution_map > 0) & (contribution_map >= con_thr)
    mismatch = np.full_like(source_map, np.nan, dtype=float)
    mismatch[study_mask] = 0.0
    mismatch[src_hot & ~con_hot] = 1.0
    mismatch[~src_hot & con_hot] = 2.0
    mismatch[src_hot & con_hot] = 3.0
    return src_hot, con_hot, mismatch, src_thr, con_thr


def _hotspot_stats(src_hot: np.ndarray, con_hot: np.ndarray, source_map: np.ndarray, contribution_map: np.ndarray) -> pd.DataFrame:
    overlap = src_hot & con_hot
    union = src_hot | con_hot
    return pd.DataFrame(
        [
            {
                "source_hotspot_area": int(np.sum(src_hot)),
                "contribution_hotspot_area": int(np.sum(con_hot)),
                "overlap_area": int(np.sum(overlap)),
                "overlap_ratio": float(np.sum(overlap) / (np.sum(union) + 1e-12)),
                "source_only_area": int(np.sum(src_hot & ~con_hot)),
                "contribution_only_area": int(np.sum(con_hot & ~src_hot)),
                "source_hotspot_sum": float(np.nansum(source_map[src_hot])),
                "contribution_hotspot_sum": float(np.nansum(contribution_map[con_hot])),
            }
        ]
    )


def _plot_panel(ax, arr: np.ndarray, study_mask: np.ndarray, outlet_code: str, cmap: str, title: str, cbar_label: str, power: bool = False):
    plot_arr = np.where(study_mask, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), np.nan)
    valid = plot_arr[np.isfinite(plot_arr) & (plot_arr > 0)]
    vmax = float(np.nanpercentile(valid, 98)) if valid.size else 1.0
    norm = colors.PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax) if power and valid.size else None
    image = ax.imshow(
        np.ma.masked_invalid(plot_arr),
        cmap=cmap,
        vmin=None if norm is not None else 0.0,
        vmax=None if norm is not None else vmax,
        norm=norm,
        interpolation="nearest",
    )
    ax.contour(study_mask.astype(float), levels=[0.5], colors="#7c8ea1", linewidths=1.0)
    oi = int(outlet_code[:4])
    oj = int(outlet_code[4:])
    ax.scatter([oj], [oi], marker="*", s=52, color="#111111", edgecolors="white", linewidths=0.45)
    ax.set_title(title)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    cb = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label)
    style_axes(ax, grid=False)


def main() -> None:
    spatial = build_full_model_spatial_data(force=True)
    source_map = np.asarray(spatial["annual_source_map"], dtype=float)
    contribution_map = np.asarray(spatial["annual_cell_contribution_map"], dtype=float)
    study_mask = np.asarray(spatial["study_area_mask"], dtype=bool)
    outlet_code = str(spatial["outlet_code"])
    data = load_tp_daily_data()
    route_mask = np.zeros_like(study_mask, dtype=bool)
    stack = [outlet_code]
    while stack:
        node = stack.pop()
        i = int(node[:4])
        j = int(node[4:])
        if 0 <= i < route_mask.shape[0] and 0 <= j < route_mask.shape[1] and not route_mask[i, j]:
            route_mask[i, j] = True
            stack.extend(str(up) for up in data["flowdir_up"].get(node, []))
    local_output_map = _build_local_output_proxy(
        source_map=source_map,
        flow_acc=np.asarray(data["grids"]["flow_acc"], dtype=float),
        dist_to_stream=np.asarray(data["grids"]["dist_to_stream"], dtype=float),
        stream_mask=np.asarray(data["grids"]["stream_mask"], dtype=float),
        landuse=np.asarray(data["landuse"], dtype=int),
        route_mask=route_mask,
        annual_total=float(np.nansum(contribution_map)),
    )
    flux_map, dropped_cycle_edges = _accumulate_routed_output(local_output_map, data["flowdir_up"], route_mask, outlet_code)

    np.save(SOURCE_MAP_NPY, source_map)
    np.save(CONTRIB_MAP_NPY, contribution_map)
    np.save(ACC_MAP_NPY, flux_map)

    landuse = pd.read_csv(Path(__file__).resolve().parents[2] / "input" / "landuse" / "mini_land_use_data.csv").to_numpy(dtype=int)
    source_diag = _source_diagnostics(source_map, landuse)
    source_export_diag = _source_export_diagnostics(source_map, contribution_map, flux_map)
    src_hot, con_hot, mismatch, src_thr, con_thr = _hotspot_masks(source_map, flux_map, study_mask)
    hotspot_stats = _hotspot_stats(src_hot, con_hot, source_map, flux_map)

    save_table(source_diag, "tp_source_map_diagnostics.csv")
    save_table(source_export_diag, "tp_source_export_map_diagnostics.csv")
    save_table(hotspot_stats, "tp_source_contribution_hotspot_statistics.csv")

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "annual_source_total": float(np.nansum(source_map)),
                "annual_cell_contribution_total": float(np.nansum(contribution_map)),
                "annual_accumulated_flux_total": float(np.nansum(flux_map)),
                "source_hotspot_threshold": src_thr,
                "contribution_hotspot_threshold": con_thr,
                "used_proxy": bool(spatial["used_proxy"]),
                "dropped_cycle_edges_for_accumulation": int(dropped_cycle_edges),
                "source_conditioning_mode": str(spatial.get("source_conditioning_mode", "unknown")),
                "raw_annual_source_total": float(spatial.get("raw_annual_source_total", np.nansum(source_map))),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 6.0), dpi=300)
    ax2, ax3 = axes.ravel()
    _plot_panel(ax2, flux_map, study_mask, outlet_code, "viridis", "Annual accumulated TP output load", f"TP output load ({TP_SPATIAL_UNIT})")
    add_panel_label(ax2, "(a)")

    cmap = plt.get_cmap("Set2", 4)
    im = ax3.imshow(np.ma.masked_invalid(mismatch), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax3.contour(study_mask.astype(float), levels=[0.5], colors="#7c8ea1", linewidths=1.0)
    oi = int(outlet_code[:4])
    oj = int(outlet_code[4:])
    ax3.scatter([oj], [oi], marker="*", s=52, color="#111111", edgecolors="white", linewidths=0.45)
    ax3.set_title("Source-output hotspot mismatch")
    ax3.set_xlabel("Column index")
    ax3.set_ylabel("Row index")
    style_axes(ax3, grid=False)
    cb = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.03)
    cb.set_ticks([0.375, 1.125, 1.875, 2.625])
    cb.set_ticklabels(["Non-hotspot", "Source hotspot only", "Contribution hotspot only", "Both hotspots"])
    add_panel_label(ax3, "(b)")
    fig.tight_layout()
    save_figure(fig, "figure_tp_source_vs_export_hotspots.png")
    plt.close(fig)

    apply_style()
    fig2, ax = plt.subplots(figsize=(7.8, 6.2), dpi=300)
    _plot_panel(ax, flux_map, study_mask, outlet_code, "magma", "Annual accumulated TP flux along routing pathways", f"Accumulated TP flux ({TP_SPATIAL_UNIT})")
    fig2.tight_layout()
    save_figure(fig2, "figure_tp_accumulated_flux_pathway.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
