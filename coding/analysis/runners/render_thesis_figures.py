"""Render standardized thesis figures from artifact plot-table specs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..reporting.thesis_figures.data import FigureBuildContext
from ..reporting.thesis_figures.suite import render_figures

log = logging.getLogger(__name__)


def _parse_compare(values: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--compare-run expects <spec>=<run-root>, got {item!r}")
        spec, root = item.split("=", 1)
        out[spec.strip()] = Path(root.strip())
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--figure", default="all", help="'all' or one figure id")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--mode", choices=("artifact", "curated"), default="curated")
    parser.add_argument("--compare-run", action="append", default=None,
                        help="Optional robustness run as <spec>=<run-root>; repeatable")
    parser.add_argument("--stats-dir", type=Path, default=None)
    parser.add_argument("--as-horizon-csv", type=Path, default=None)
    parser.add_argument("--size-grid-root", type=Path, default=None)
    parser.add_argument("--volume-bucket-csv", type=Path, default=None)
    parser.add_argument("--volume-by-date-csv", type=Path, default=None)
    parser.add_argument("--volume-db", type=Path, default=None)
    parser.add_argument("--tier-map-csv", type=Path, default=None)
    parser.add_argument("--membership-root", type=Path, default=None)
    parser.add_argument(
        "--export-close-share-xlsx",
        nargs="?",
        const="",
        default=None,
        help=(
            "Export the Figure 7.2 close-share audit workbook. "
            "Without a path, writes closing_auction_share_daily_values.xlsx to --out."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    default_out = Path(__file__).resolve().parents[3] / "thesis" / "figures" / "final_20260618_queue"
    out_dir = args.out or default_out
    if args.export_close_share_xlsx is None:
        close_share_xlsx_path = None
    elif args.export_close_share_xlsx == "":
        close_share_xlsx_path = out_dir / "closing_auction_share_daily_values.xlsx"
    else:
        close_share_xlsx_path = Path(args.export_close_share_xlsx)
    ctx = FigureBuildContext(
        run_root=args.run_root,
        out_dir=out_dir,
        compare_runs=_parse_compare(args.compare_run),
        stats_dir=args.stats_dir,
        as_horizon_csv=args.as_horizon_csv,
        size_grid_root=args.size_grid_root,
        volume_bucket_csv=args.volume_bucket_csv,
        volume_by_date_csv=args.volume_by_date_csv,
        volume_db=args.volume_db,
        tier_map_csv=args.tier_map_csv,
        membership_root=args.membership_root,
        close_share_xlsx_path=close_share_xlsx_path,
    )
    manifest = render_figures(
        ctx,
        figure=args.figure,
        mode=args.mode,
        refresh_data=args.refresh_data,
        fail_fast=args.fail_fast,
    )
    log.info(
        "Rendered %d figures to %s (%d skipped)",
        len(manifest["outputs"]),
        ctx.out_dir,
        len(manifest["skipped"]),
    )
    if manifest["skipped"]:
        for fig, reason in manifest["skipped"].items():
            log.warning("%s skipped: %s", fig, reason)


if __name__ == "__main__":
    main()
