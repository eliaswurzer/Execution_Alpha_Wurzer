"""Classify missing expected-closing-volume symbol-days.

The audit is intentionally read-only with respect to simulation artifacts.  It
explains whether ``missing_expected_vc`` rows can be recovered from causal
same-symbol history, from an approved predecessor ticker mapping, or should
remain excluded under the conservative headline policy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from ..utils.symbols import canonical_symbol
from . import master_panel


DEFAULT_MEMBERSHIP = (
    Path(__file__).resolve().parents[3]
    / "reference"
    / "index_membership"
    / "sp500_membership_intervals.csv"
)


def _date(value) -> dt.date:
    return pd.Timestamp(value).date()


def _load_membership(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_MEMBERSHIP
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "effective_from", "effective_to"])
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"symbol", "effective_from", "effective_to"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Membership file {path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].map(canonical_symbol)
    frame["effective_from"] = pd.to_datetime(frame["effective_from"]).dt.date
    frame["effective_to"] = pd.to_datetime(frame["effective_to"]).dt.date
    return frame


def _is_active(symbol: str, date: dt.date, membership: pd.DataFrame) -> bool:
    if membership.empty:
        return True
    rows = membership[membership["symbol"] == canonical_symbol(symbol)]
    if rows.empty:
        return True
    return bool(((rows["effective_from"] <= date) & (date <= rows["effective_to"])).any())


def _history_count(vc_history: pd.DataFrame, symbol: str, before: dt.date) -> int:
    if vc_history.empty:
        return 0
    return int(
        (
            (vc_history["symbol"] == canonical_symbol(symbol))
            & (vc_history["date"] < before)
            & pd.to_numeric(vc_history["vc_shares"], errors="coerce").gt(0)
        ).sum()
    )


def classify_missing_expected_vc(
    failures: pd.DataFrame,
    vc_history: pd.DataFrame,
    identity_map: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    if failures.empty:
        return pd.DataFrame(columns=[
            "date", "symbol", "recovery_class", "source_symbol",
            "mapping_type", "headline_allowed", "same_symbol_history_rows",
            "predecessor_history_rows", "detail",
        ])

    vc = vc_history.copy()
    if not vc.empty:
        vc["symbol"] = vc["symbol"].map(canonical_symbol)
        vc["date"] = pd.to_datetime(vc["date"]).dt.date

    rows: list[dict] = []
    missing = failures[failures["reason"] == "missing_expected_vc"].copy()
    for item in missing.itertuples(index=False):
        symbol = canonical_symbol(getattr(item, "symbol"))
        date = _date(getattr(item, "date"))
        same_rows = _history_count(vc, symbol, date)
        maps = identity_map[identity_map["target_symbol"] == symbol]
        active = _is_active(symbol, date, membership)

        recovery_class = "unrecoverable_no_history"
        source_symbol = ""
        mapping_type = ""
        headline_allowed = False
        predecessor_rows = 0
        detail = "No causal same-symbol or approved predecessor history was found."

        if not active:
            recovery_class = "excluded_no_regular_close"
            detail = "The current membership interval file excludes this symbol-day."
        elif same_rows >= 5:
            recovery_class = "recoverable_same_symbol_history"
            detail = "At least five prior same-symbol auction-volume observations exist."
        elif not maps.empty:
            approved = maps[maps["headline_allowed"] & maps["source_symbol"].astype(bool)]
            if not approved.empty:
                row = approved.iloc[0]
                source_symbol = str(row["source_symbol"])
                mapping_type = str(row["mapping_type"])
                headline_allowed = True
                cutoff = row["effective_from"] if pd.notna(row["effective_from"]) else date
                predecessor_rows = _history_count(vc, source_symbol, min(date, cutoff))
                if predecessor_rows >= 5:
                    recovery_class = "recoverable_predecessor_mapping"
                    detail = (
                        "Approved predecessor mapping and sufficient causal "
                        "predecessor VC history are available."
                    )
                else:
                    recovery_class = "unrecoverable_no_history"
                    detail = (
                        "Approved predecessor mapping exists, but fewer than "
                        "five prior predecessor VC observations were found."
                    )
            else:
                row = maps.iloc[0]
                source_symbol = str(row["source_symbol"])
                mapping_type = str(row["mapping_type"])
                headline_allowed = False
                recovery_class = "ambiguous_corporate_action"
                detail = (
                    "Identity map marks this symbol as unsuitable for headline "
                    "backfill without a verified scaling rule."
                )

        rows.append({
            "date": date,
            "symbol": symbol,
            "recovery_class": recovery_class,
            "source_symbol": source_symbol,
            "mapping_type": mapping_type,
            "headline_allowed": headline_allowed,
            "same_symbol_history_rows": same_rows,
            "predecessor_history_rows": predecessor_rows,
            "detail": detail,
        })
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def audit_run(
    run_root: Path,
    *,
    out_dir: Path | None = None,
    identity_map_path: Path | None = None,
    membership_path: Path | None = None,
) -> tuple[Path, Path]:
    run_root = Path(run_root)
    out = out_dir or (run_root / "metadata")
    identity_map_path = identity_map_path or master_panel.EXPECTED_VC_IDENTITY_MAP
    membership_path = membership_path or DEFAULT_MEMBERSHIP
    failures_path = run_root / "metadata" / "simulation_failures.csv"
    vc_path = run_root / "cache" / "vc_history.parquet"
    if not failures_path.exists():
        raise FileNotFoundError(f"Missing simulation failures CSV: {failures_path}")
    failures = pd.read_csv(failures_path)
    vc_history = pd.read_parquet(vc_path) if vc_path.exists() else pd.DataFrame()
    identity_map = master_panel._load_expected_vc_identity_map(identity_map_path)
    membership = _load_membership(membership_path)
    detail = classify_missing_expected_vc(
        failures, vc_history, identity_map, membership,
    )
    out.mkdir(parents=True, exist_ok=True)
    detail_path = out / "expected_vc_recovery_audit.csv"
    summary_path = out / "expected_vc_recovery_audit_summary.json"
    detail.to_csv(detail_path, index=False)
    counts = (
        detail["recovery_class"].value_counts(dropna=False).sort_index().to_dict()
        if not detail.empty else {}
    )
    summary_path.write_text(
        json.dumps({
            "run_root": str(run_root),
            "expected_vc_policy_version": master_panel.EXPECTED_VC_POLICY_VERSION,
            "identity_map_path": str(identity_map_path),
            "identity_map_sha256": master_panel._sha256(identity_map_path)
            if identity_map_path.exists() else None,
            "membership_path": str(membership_path),
            "missing_expected_vc_rows": int(len(detail)),
            "recovery_class_counts": {str(k): int(v) for k, v in counts.items()},
        }, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return detail_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--identity-map", type=Path, default=master_panel.EXPECTED_VC_IDENTITY_MAP)
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    args = parser.parse_args()
    detail, summary = audit_run(
        args.run_root,
        out_dir=args.out_dir,
        identity_map_path=args.identity_map,
        membership_path=args.membership,
    )
    print(f"Wrote {detail}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
