"""Static posting-distance diagnostics for thesis figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def posting_curve_summary(
    candidate_panel: pd.DataFrame,
    *,
    offset_col: str = "limit_offset_bps",
    fill_col: str = "event",
    value_col: str = "target_net_alpha_vs_moc_bps",
    group_cols: tuple[str, ...] = ("side", "tier"),
) -> pd.DataFrame:
    """Aggregate fill probability and value by posting distance."""
    required = {offset_col, fill_col, value_col, *group_cols}
    missing = required - set(candidate_panel.columns)
    if missing:
        raise ValueError(f"Candidate panel missing columns: {sorted(missing)}")
    df = candidate_panel.copy()
    df[offset_col] = pd.to_numeric(df[offset_col], errors="coerce")
    df[fill_col] = pd.to_numeric(df[fill_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[offset_col, fill_col, value_col])
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, offset_col, "n", "fill_probability", "mean_value_bps"])
    out = (
        df.groupby([*group_cols, offset_col], dropna=False)
        .agg(
            n=(fill_col, "size"),
            fill_probability=(fill_col, "mean"),
            mean_value_bps=(value_col, "mean"),
            median_value_bps=(value_col, "median"),
        )
        .reset_index()
        .sort_values([*group_cols, offset_col])
        .reset_index(drop=True)
    )
    out["fill_probability"] = out["fill_probability"].clip(0.0, 1.0)
    return out


def save_posting_curve_figure(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    offset_col: str = "limit_offset_bps",
) -> Path:
    """Save a two-panel posting-distance figure."""
    if summary.empty:
        raise ValueError("Cannot plot an empty posting-curve summary")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    group_cols = [c for c in ("side", "tier") if c in summary.columns]
    iterator = summary.groupby(group_cols, dropna=False) if group_cols else [("all", summary)]
    for key, grp in iterator:
        label = key if isinstance(key, str) else ", ".join(str(x) for x in key)
        axes[0].plot(grp[offset_col], grp["fill_probability"], marker="o", label=label)
        axes[1].plot(grp[offset_col], grp["mean_value_bps"], marker="o", label=label)
    axes[0].set_ylabel("Execution probability")
    axes[1].set_ylabel("Mean net alpha vs MOC, bps")
    for ax in axes:
        ax.set_xlabel("Posting distance from touch, bps")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path