"""Causal rolling-window helpers for value-aware model training."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from .. import config as cfg


@dataclass(frozen=True)
class RollingTrainingWindow:
    anchor_date: dt.date
    train_start: dt.date
    train_end: dt.date
    n_train_dates: int
    lookback_days: int
    min_lookback_days: int


def monthly_anchor_dates(dates: list[dt.date]) -> list[dt.date]:
    """Return the first available trading date of each calendar month."""
    if not dates:
        return []
    frame = pd.DataFrame({"date": sorted(pd.Timestamp(d).date() for d in dates)})
    frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M")
    return [pd.Timestamp(x).date() for x in frame.groupby("month")["date"].min()]


def rolling_training_window(
    anchor_date: dt.date,
    available_dates: list[dt.date],
    *,
    lookback_days: int = cfg.VALUE_MODEL_LOOKBACK_DAYS,
    min_lookback_days: int = cfg.VALUE_MODEL_MIN_LOOKBACK_DAYS,
) -> RollingTrainingWindow | None:
    """Build a causal training window ending before ``anchor_date``.

    The current anchor date is excluded. ``None`` means the anchor is still in
    the warm-up period and should not produce rolling headline predictions.
    """
    prior = [d for d in sorted(available_dates) if d < anchor_date]
    if len(prior) < int(min_lookback_days):
        return None
    train = prior[-int(lookback_days):]
    return RollingTrainingWindow(
        anchor_date=anchor_date,
        train_start=train[0],
        train_end=train[-1],
        n_train_dates=len(train),
        lookback_days=int(lookback_days),
        min_lookback_days=int(min_lookback_days),
    )


def build_monthly_training_schedule(
    dates: list[dt.date],
    *,
    lookback_days: int = cfg.VALUE_MODEL_LOOKBACK_DAYS,
    min_lookback_days: int = cfg.VALUE_MODEL_MIN_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Return monthly anchors with causal train windows after warm-up."""
    rows = []
    for anchor in monthly_anchor_dates(dates):
        window = rolling_training_window(
            anchor,
            dates,
            lookback_days=lookback_days,
            min_lookback_days=min_lookback_days,
        )
        if window is None:
            rows.append({
                "anchor_date": anchor,
                "status": "warmup_excluded",
                "train_start": pd.NaT,
                "train_end": pd.NaT,
                "n_train_dates": 0,
            })
        else:
            rows.append({
                "anchor_date": window.anchor_date,
                "status": "trainable",
                "train_start": window.train_start,
                "train_end": window.train_end,
                "n_train_dates": window.n_train_dates,
            })
    return pd.DataFrame(rows)


def assign_monthly_anchor(
    evaluation_dates: list[dt.date],
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Map each evaluation date to the most recent trainable monthly anchor."""
    if schedule.empty:
        return pd.DataFrame(columns=["date", "anchor_date", "status"])
    anchors = schedule[schedule["status"] == "trainable"].copy()
    if anchors.empty:
        return pd.DataFrame({
            "date": sorted(evaluation_dates),
            "anchor_date": pd.NaT,
            "status": "warmup_excluded",
        })
    anchors["anchor_date"] = pd.to_datetime(anchors["anchor_date"])
    rows = []
    for date in sorted(evaluation_dates):
        eligible = anchors[anchors["anchor_date"].dt.date <= date]
        if eligible.empty:
            rows.append({"date": date, "anchor_date": pd.NaT, "status": "warmup_excluded"})
        else:
            rows.append({
                "date": date,
                "anchor_date": pd.Timestamp(eligible["anchor_date"].max()).date(),
                "status": "mapped",
            })
    return pd.DataFrame(rows)