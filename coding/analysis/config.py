"""
config.py -- Zentrale Konstanten und Parameter fuer die gesamte Analyse-Pipeline.

Alle Werte entsprechen der Thesis-Spezifikation; Abweichungen sind mit
``# THESIS_DEVIATION`` markiert. Import-Konvention::

    from analysis import config as cfg
"""

from __future__ import annotations

import datetime as _dt
import os as _os
from pathlib import Path

from .data import trade_conditions as _tc

# ---------------------------------------------------------------------------
# Pfade — env-var overrides allow running on machines where data lives outside
# the repo tree (e.g. a separate analysis drive).  Set THESIS_DATA_ROOT and
# THESIS_ARTIFACTS_DIR before invoking any runner.
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
RUN_ROOT: Path = Path(_os.environ.get(
    "THESIS_RUN_ROOT",
    str(Path.home() / "Documents" / "master thesis"),
))
DATA_ROOT: Path = Path(_os.environ.get("THESIS_DATA_ROOT", str(RUN_ROOT / "data")))
ARTIFACTS_DIR: Path = Path(_os.environ.get("THESIS_ARTIFACTS_DIR", str(RUN_ROOT / "artifacts")))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# TAQ Parquet root. The 2018/2019 streaming preprocessing outputs were
# consolidated into one root (preprocessing/consolidate_processed_roots.py,
# see consolidation_manifest.json there); the loader resolves per-date
# directories, so both years share the same root.
# THESIS_TAQ_PARQUET_<YEAR> can still point a single year elsewhere.
CONSOLIDATED_TAQ_ROOT: Path = DATA_ROOT / "sp500_preprocess_2018_2019"


def _default_year_parquet_dir(year: int) -> Path:
    if CONSOLIDATED_TAQ_ROOT.exists():
        return CONSOLIDATED_TAQ_ROOT
    return DATA_ROOT / str(year)


def _year_parquet_dir(year: int) -> Path:
    return Path(_os.environ.get(
        f"THESIS_TAQ_PARQUET_{year}",
        str(_default_year_parquet_dir(year)),
    ))


TAQ_PARQUET_DIR = {
    2018: _year_parquet_dir(2018),
    2019: _year_parquet_dir(2019),
}

# Warn at import time if key paths are missing so errors surface early rather
# than deep inside worker processes with confusing tracebacks.
def _warn_missing(label: str, path: Path) -> None:
    import logging as _logging
    if not path.exists():
        _logging.getLogger(__name__).warning(
            "config: %s does not exist: %s  "
            "(set THESIS_DATA_ROOT, THESIS_TAQ_PARQUET_<YEAR>, or "
            "THESIS_ARTIFACTS_DIR env vars if paths differ)",
            label, path,
        )

_warn_missing("DATA_ROOT", DATA_ROOT)
_warn_missing("TAQ_PARQUET_DIR[2018]", TAQ_PARQUET_DIR[2018])
_warn_missing("TAQ_PARQUET_DIR[2019]", TAQ_PARQUET_DIR[2019])

# volume/-DuckDB mit Dollar-Volumen je (Ticker, Date, Bucket)
VOLUME_DB_PATH: Path = Path(_os.environ.get(
    "THESIS_VOLUME_DB",
    str(RUN_ROOT / "volume" / "dollar_volume.duckdb"),
))
INDEX_MEMBERSHIP_DIR: Path = Path(_os.environ.get(
    "THESIS_INDEX_MEMBERSHIP_DIR",
    str(REPO_ROOT / "reference" / "index_membership"),
))


# ---------------------------------------------------------------------------
# Trading-Session
# ---------------------------------------------------------------------------

RTH_OPEN = _dt.time(9, 30)
RTH_CLOSE = _dt.time(16, 0)
# Closing-Auction-Prints werden in TAQ einige Sekunden bis Minuten nach
# 16:00:00 disseminiert. Loader nutzt dieses erweiterte Cutoff um sie zu
# behalten; Strategie- und Feature-Logik beachten weiterhin RTH_CLOSE.
RTH_LOAD_CUTOFF = _dt.time(16, 5)
MOC_CUTOFF = _dt.time(15, 50)  # T1 -- NYSE Rule 7.35A
# Daily TAQ closing-auction prints can be disseminated several minutes after
# 16:00.  Evaluation features still use the RTH load cutoff above; the auction
# extractor searches this wider post-close window for ``6``/``M`` markers.
CLOSING_AUCTION_SEARCH_END = _dt.time(16, 15)
# Documented evaluation-calendar exclusions. Early-close sessions (13:00 ET
# close: July 3, day after Thanksgiving, Christmas Eve) have no 15:30 arrival
# window, no 15:50 MOC cutoff, and no 16:00 closing auction, so the thesis
# design is undefined on those days; excluding half days is standard practice
# in intraday execution studies. 2019-05-13 is excluded because the raw TAQ
# trade file is unobtainable (see data-availability audit).
EXCLUDED_EVAL_DATES: frozenset = frozenset({
    _dt.date(2018, 7, 3),    # early close 13:00
    _dt.date(2018, 11, 23),  # early close 13:00
    _dt.date(2018, 12, 24),  # early close 13:00
    _dt.date(2019, 5, 13),   # raw trade file unobtainable
    _dt.date(2019, 7, 3),    # early close 13:00
    _dt.date(2019, 11, 29),  # early close 13:00
    _dt.date(2019, 12, 24),  # early close 13:00
})

# NYSE Imbalance-Dissemination beginnt ca. 15:50; Nasdaq ca. 15:55. Wird in
# H1 als pre/post-dissemination Subgroup-Filter genutzt.
DISSEMINATION_START_BY_LISTING: dict[str, _dt.time] = {
    "NYSE": _dt.time(15, 50),
    "NASDAQ": _dt.time(15, 55),
}

# Trade-Signing-Methode (siehe microstructure/signing.py)
# Thesis §4.2: Headline = Holden-Jacobsen-corrected Lee-Ready (millisecond-Anpassung
# fuer DTAQ); Plain Lee-Ready (1991) ist Robustness-Spezifikation.
TRADE_SIGN_METHOD: str = "holden_jacobsen"   # "lee_ready" | "holden_jacobsen"
HJ_LAG_MS: int = 1                            # Lag fuer Holden-Jacobsen-Quote-Lookup
AUCTION_IMBALANCE_START = _dt.time(15, 50)  # official dissemination/subgroup timing

# Public pre-cutoff imbalance proxy used by S3_IMB/S3_FULL. This is not the
# official auction-imbalance feed; it is a causal NBBO/OFI pressure proxy that
# is available before the MOC cutoff and can therefore affect limit placement.
IMBALANCE_PROXY_START = _dt.time(15, 30)
IMBALANCE_PROXY_LOOKBACK_SECONDS = 600


# ---------------------------------------------------------------------------
# Strategie-Parameter
# ---------------------------------------------------------------------------

# Refresh-Intervall der passiven Strategien (Sekunden). Thesis default 30s.
REFRESH_SECONDS_DEFAULT = 30
REFRESH_SECONDS_ROBUSTNESS = (15, 30, 60)

# Ausfuehrungs-Fenster A/B/C mit Startzeiten.
# Thesis §5.1: Baseline = 30 min vor close; Robustness 15 min und 60 min.
# A = 15:00 (60 min, robustness long); B = 15:30 (30 min, headline);
# C = 15:45 (15 min, robustness short).
EXECUTION_WINDOWS: dict[str, _dt.time] = {
    "A": _dt.time(15, 0),
    "B": _dt.time(15, 30),  # Primary Window (30 min Baseline -- Thesis Headline)
    "C": _dt.time(15, 45),
}
PRIMARY_WINDOW = "B"

# Max passiver Offset (delta_max) in Basispunkten je Liquiditaets-Tier.
# Tier 1 = enge Spreads (liquide), Tier 3 = weite Spreads.
# Initialwerte: werden in Phase C/D durch Grid-Search auf Pre-Sample ueberschrieben.
DELTA_MAX_BPS: dict[int, float] = {1: 2.0, 2: 5.0, 3: 10.0}

# Vol-Regime-Schwelle aus Thesis Eq. nach 4.14: hoch-vol wenn sigma_t > c * sigma_bar
VOL_REGIME_THRESHOLD = 1.5
VOL_REGIME_HIGH_MULTIPLIER = 1.2  # breitet delta_max aus
VOL_REGIME_LOW_MULTIPLIER = 0.8   # bringt Order naeher an den Touch

# Clipping-Bounds fuer S3-Signalfaktoren
F_OFI_MIN = 0.3       # verhindert Spread-Cross
F_IMB_MIN = 0.5
F_IMB_MAX = 1.5

# Max Slice = 5 % des Trailing-20-Tage-Avg-Closing-Auktions-Volumen
MAX_SLICE_FRACTION_OF_VC = 0.05

# Tick-Diskretisierung der Limit-Preise. Reg NMS Rule 612 verbietet
# Sub-Penny-Limits fuer Aktien >= $1; das Universum filtert ohnehin auf
# >= $5. Snapping erfolgt in die passive Richtung (BUY floor, SELL ceil),
# damit der gesnappte Preis nie aggressiver ist als der Modell-Offset.
TICK_SIZE = 0.01
SNAP_LIMIT_TO_TICK = True

# Daily-TAQ-NBBO-Sizes sind in Round Lots denominiert (1 Lot = 100 Shares);
# Trade-Volumina sind in Shares. Verifiziert an den preprocessten Parquets
# (AAPL 2018-01-03: Median best_bid_size = 4 Lots vs. Median Trade = 100
# Shares). Queue-Ahead-Schaetzungen muessen daher mit diesem Faktor in
# Shares umgerechnet werden.
NBBO_SIZE_SHARES_PER_LOT = 100


# ---------------------------------------------------------------------------
# Parent-Order-Sizing
# ---------------------------------------------------------------------------

# Parent-Order-Groessen als Fraktion des erwarteten Auktionsvolumens E[V_C].
# Thesis §5.1: grid (0.5%, 1%, 2%, 5%, 10%) mit Headline = 1%.
PARENT_ORDER_SIZE_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)
PARENT_ORDER_PRIMARY_FRACTION = 0.01  # Thesis-Headline; groessere Fraktionen aktivieren den self-impact Term


# ---------------------------------------------------------------------------
# Feature-Konstruktion
# ---------------------------------------------------------------------------

# Bump this whenever feature timestamps, z-scoring, or other simulation-visible
# state construction semantics change. Calibration artifacts and panel shards
# with an older value are intentionally invalid for headline evaluation.
# v2: q0/D0 state covariates and the auction-imbalance proxy converted from
#     Daily-TAQ round lots to shares (x NBBO_SIZE_SHARES_PER_LOT).
FEATURE_POLICY_VERSION = "causal_features_v2"

# 5-min Realised Volatility = Summe ueber 60 x 5s log-returns
# (Thesis §3.6 "regime function" v(.) und §5.2 Fill-Model state vector)
RV_WINDOW_SECONDS = 300
RV_SUB_INTERVAL_SECONDS = 5

# OFI-Aggregation (Thesis §3.5, eq:ofi). Bucket = 30 s ist angepasst an
# (a) Cont-Kukanov-Stoikov (2014) Standard-Tests (0.5-30 s),
# (b) die Strategie-Refresh-Kadenz REFRESH_SECONDS_DEFAULT = 30 s, so dass jede
# Strategie-Entscheidung einen frischen OFI-Wert sieht statt einen veralteten,
# (c) die Adverse-Selection-Horizont AS_HORIZON_SECONDS = 30 s, was die
# pre-fill und post-fill Microstructure-Signale auf eine gemeinsame Skala
# bringt. 30 min Fenster / 30 s Bucket = 60 Buckets pro Symbol-Tag, das
# ausreicht fuer einen stabilen within-symbol-day Z-Score.
# Robustness-Grid (10s, 60s) wird in der Robustness-Section ausgewiesen.
OFI_WINDOW_SECONDS = 30
OFI_WINDOW_GRID_SECONDS: tuple[int, ...] = (10, 30, 60)
OFI_ZSCORE_WINDOW_BUCKETS = 60  # legacy parameter, wird nur in zscore_mode="rolling" beruecksichtigt

# Time-of-Day Bins (Stunden) als kategorische Dummies im State-Vektor
# Hours explicitly encoded as one-hot dummies in the state vector.
# Hour 9 (09:30–10:00) is the implicit baseline (all dummies = 0) and is NOT
# listed here — this is intentional, not an omission.
TOD_HOUR_BINS = (10, 11, 12, 13, 14, 15)


# ---------------------------------------------------------------------------
# Fill-Modell
# ---------------------------------------------------------------------------

# Legacy 5-minute horizon (no longer used for fill-model calibration; kept only
# for backward-compatible references).
FILL_HORIZON_MINUTES = 5
# Fill-model calibration / query horizon. The simulation re-prices every
# REFRESH_SECONDS_DEFAULT seconds and does NOT hold an unfilled slice as a
# standing order, so a single posting effectively lives one refresh interval.
# Calibrating, censoring, and querying the survival model all at this same
# horizon keeps the fitted curve aligned with where it is evaluated, and makes
# the 30-second sampling grid non-overlapping (window == grid step), restoring
# observation independence for the partial-likelihood / Cox objective.
FILL_MODEL_HORIZON_SECONDS = REFRESH_SECONDS_DEFAULT
# Passive fill calibration samples synthetic limit orders at several distances
# from the touch because TAQ/NBBO has no full depth or queue-position data.
# 0 bps is an at-touch order; larger values model deeper passive posting.
FILL_MODEL_OFFSET_GRID_BPS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)

# Value-aware execution model. This layer is deliberately separate from the
# headline S0-S4 strategies. It trains on realized net alpha versus MOC, not
# only on fill/no-fill outcomes, and is evaluated after a causal warm-up.
# v2: target follows the implementation-shortfall net-alpha convention. The
#     close-relative gross term already contains realized post-fill drift, so
#     adverse selection is diagnostic and is not deducted a second time.
VALUE_MODEL_POLICY_VERSION = "rolling_value_model_v2"
VALUE_MODEL_LOOKBACK_DAYS = 120
VALUE_MODEL_MIN_LOOKBACK_DAYS = 60
VALUE_MODEL_ANCHOR_FREQUENCY = "monthly"
VALUE_MODEL_OFFSET_GRID_BPS: tuple[float, ...] = FILL_MODEL_OFFSET_GRID_BPS
VALUE_MODEL_MIN_EXPECTED_ALPHA_BPS = 0.0
VALUE_MODEL_MIN_ROWS_GLOBAL = 100
VALUE_MODEL_MIN_ROWS_SIDE = 50
VALUE_MODEL_MIN_ROWS_SIDE_TIER = 25
VALUE_MODEL_MIN_TARGET_STD_BPS = 1e-8

# Pre-Sample und Evaluations-Window (Thesis §4.4/§4.5, final design):
# Kalibrierung = H1-2018, Evaluation = H2-2018 + 2019. Das konsolidierte
# 2018-2019-Archiv ist vollstaendig bis auf 2019-05-13 (Rohdatei nicht
# beschaffbar; dokumentierter Einzelausschluss, siehe Audit/Membership-Check).
PRE_SAMPLE_START = _dt.date(2018, 1, 2)
PRE_SAMPLE_END = _dt.date(2018, 6, 29)
EVAL_START = _dt.date(2018, 7, 2)
EVAL_END = _dt.date(2019, 12, 31)

# Adverse-Selection Horizont (Thesis §3.4 + §5.2: Headline = 30s).
# ``THESIS_AS_HORIZON_SECONDS`` is only for isolated robustness reruns. The
# active value is included in the master-panel fingerprint, so shards from
# different markout horizons cannot be resumed into one another.
AS_HEADLINE_HORIZON_SECONDS = 30                     # explizit, fuer Reporting
AS_HORIZON_SECONDS = int(_os.environ.get(             # Pipeline-Default == Headline
    "THESIS_AS_HORIZON_SECONDS",
    str(AS_HEADLINE_HORIZON_SECONDS),
))
AS_HORIZON_GRID_SECONDS: tuple[int, ...] = (5, 15, 30, 60, 300)


# ---------------------------------------------------------------------------
# Transaktionskosten (Thesis §3.2, decomposition eq:alpha_decomp,
# Komponenten alpha_rebate / alpha_fees)
# ---------------------------------------------------------------------------

# Kalibrierung fuer das Megacap-Universum (S&P 500, median share price 2018-2019
# ~80-100 USD) gegen die oeffentlichen NYSE-Arca / Nasdaq Make-Take-Fee
# Schedules:
#   * Maker-Rebate: 0.0029 USD/Share auf tape A/B/C -> 0.0029 / 100 = 0.29 bp
#   * Taker-Fee:    0.0030 USD/Share              -> 0.0030 / 100 = 0.30 bp
#   * Broker-Commission (algorithmic execution-only fee 2018-2019):
#                   ~0.0010 USD/Share              -> 0.0010 / 100 = 0.10 bp
# Konsistent mit den in Battalio, Corwin & Jennings (2016) berichteten
# Make-Take-Tarif-Niveaus fuer den 2014-2015 US-Equity-Markt; die Tarife
# blieben 2018-2019 effektiv unveraendert.
MAKER_REBATE_BPS = 0.29
TAKER_FEE_BPS = 0.30
COMMISSION_BPS = 0.10

# ---------------------------------------------------------------------------
# Market-Impact (Square-Root) -- Komponente alpha_impact
# ---------------------------------------------------------------------------
# Aktivierung: nur wenn parent_size_pct > IMPACT_ACTIVATION_THRESHOLD wird
# Impact berechnet. Modell (Square-Root in der Almgren-Chriss-Tradition,
# Almgren & Chriss 2001):
#       impact_bps = IMPACT_COEF_BPS * sqrt(parent_size_pct)
# wobei parent_size_pct = parent_qty / expected_close_volume.
# Default-Schwelle 0.01 = 1% des erwarteten Auktionsvolumens; darunter wird
# Self-Impact bei einem passiv-resting Submitter im kontinuierlichen Markt
# als vernachlaessigbar betrachtet.
#
# HINWEIS: dieser Term modelliert *self-impact eines passiv resting Submitters
# im kontinuierlichen Markt*, nicht das closing-auction Impact-Niveau aus
# Goyal et al. (2026); die beiden Regimes sind physikalisch verschieden und
# es gibt in der Literatur keinen Konsens fuer den passiv-resting-Koeffizienten.
# Der Default-Wert ist daher als Modellannahme zu verstehen, ueber die der
# IMPACT_COEF_BPS_GRID-Sweep eine explizite Robustness-Bracket bildet.
IMPACT_ACTIVATION_THRESHOLD = 0.01
IMPACT_COEF_BPS = 8.0   # Headline-Wert, Almgren-Chriss-style continuous-market self-impact
IMPACT_COEF_BPS_GRID: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0)  # Robustness-Sweep ueber kappa


# ---------------------------------------------------------------------------
# Universum und Filter (Thesis §5.2/5.3)
# ---------------------------------------------------------------------------

# Top-N nach Trailing-Dollar-ADV
UNIVERSE_TOP_N_DEFAULT = 500
UNIVERSE_ADV_LOOKBACK_DAYS = 252  # Thesis §4.4: trailing 12 trading months (~252 days)
UNIVERSE_MIN_PRICE = 5.0
UNIVERSE_MIN_TRADES_PER_DAY = 100_000  # Halt-Detection Heuristik

# NBBO / Trade cleaning
TRADE_MIDQUOTE_DEV_MAX = 0.10    # 10 % Obergrenze fuer Trade-Preis-Abweichung von Midquote
QUOTED_SPREAD_MAX_FRAC = 0.05    # 5 % des Midquote; weitere Filter fuer Stub-Quotes
# Tape-replay passive fills must be checked against displayed/lit venues only.
# In the processed Daily TAQ archive used here, exchange code "D" is a
# non-displayed/TRF-style venue bucket and cannot execute a displayed passive
# quote resting on the lit book.
TAPE_REPLAY_EXCLUDED_EXCHANGES = {"D"}
TAPE_REPLAY_VOLUME_PARTICIPATION = 1.0

# S3 OFI uses the within-symbol-day z-score from microstructure.ofi. Keeping
# this scale explicit prevents raw share counts from being damped by ADV.
OFI_SIGNAL_SCALE = 1.0

# Thesis §5.3: Sale Conditions die gedroppt werden (cancelled / corrected / OOS)
TRADE_CONDITION_POLICY_VERSION = _tc.POLICY_VERSION
TRADE_VALID_CORRECTIONS = _tc.VALID_CORRECTIONS
TRADE_BAD_CONDITIONS = _tc.EVALUATION_BAD_CONDITIONS
TRADE_PREPROCESS_BAD_CONDITIONS = _tc.PREPROCESS_BAD_CONDITIONS
EXPECTED_PREPROCESS_TRADE_FILTER_POLICY = "preprocessing"
TRADE_QC_POLICY_CHECK_MODE = _os.environ.get(
    "THESIS_TRADE_QC_POLICY_CHECK",
    "enforce",
).strip().lower()
# Daily TAQ uses ``6`` for the market-center closing trade and ``M`` for the
# market-center official close marker.  The marker supplies a close price when
# needed, but it is not a second closing-auction volume print.
CLOSING_TRADE_CONDITIONS = _tc.CLOSING_TRADE_CONDITIONS
OFFICIAL_CLOSE_CONDITIONS = _tc.OFFICIAL_CLOSE_CONDITIONS
CLOSING_AUCTION_CONDITIONS = _tc.CLOSING_AUCTION_CONDITIONS
OPENING_AUCTION_CONDITIONS = _tc.OPENING_AUCTION_CONDITIONS


# ---------------------------------------------------------------------------
# Placebo-Test (Thesis §4.5.3)
# ---------------------------------------------------------------------------

PLACEBO_WINDOW_START = _dt.time(11, 0)
PLACEBO_WINDOW_END = _dt.time(12, 0)


# ---------------------------------------------------------------------------
# Run-Defaults
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Pilot-Default-Universum
# ---------------------------------------------------------------------------
# Kompakte, repraesentative Liste S&P-500-Mega-Caps fuer Pilot- und
# Smoke-Tests, wenn kein vollstaendiges Top-N-Universum aus dem Volume-Panel
# konstruiert werden kann (z.B. 3-Tage Pilot ohne 60-Tage ADV-History).
PILOT_UNIVERSE: tuple[str, ...] = (
    "AAPL", "MSFT", "AMZN", "GOOG L", "GOOG", "FB", "NVDA", "BRK B", "TSLA",
    "JPM", "V", "JNJ", "PG", "MA", "HD", "UNH", "BAC", "DIS", "VZ", "ADBE",
    "NFLX", "CRM", "PFE", "KO", "PEP", "INTC", "T", "WMT", "MRK", "CSCO",
    "XOM", "NKE", "ORCL", "AVGO", "ABT", "QCOM", "COST", "TMO", "CVX", "MCD",
)

