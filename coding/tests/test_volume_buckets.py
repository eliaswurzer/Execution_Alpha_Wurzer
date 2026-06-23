from __future__ import annotations

import datetime as dt

import pytest

pl = pytest.importorskip("polars")

import buckets
import db as volume_db


def test_close_auction_volume_uses_delayed_closing_trade_not_official_close_marker() -> None:
    times = [
        dt.datetime(2018, 2, 1, 9, 30),
        dt.datetime(2018, 2, 1, 16, 0),
        dt.datetime(2018, 2, 1, 16, 0, 13),
    ] + [dt.datetime(2018, 2, 1, 12, 0, i) for i in range(7)]
    conditions = ["O", "M", "6"] + [""] * 7
    volumes = [1, 2, 3] + [1] * 7
    frame = pl.DataFrame({
        "time": times,
        "price": [10.0] * 10,
        "volume": volumes,
        "sale_condition": conditions,
    })

    row = buckets.compute_symbol_volume(frame.lazy(), "AAPL", dt.date(2018, 2, 1))

    assert row is not None
    assert row["Open_Auction_Val"] == 10.0
    assert row["Close_Auction_Val"] == 30.0
    assert row["Official_Close_Marker_Val"] == 20.0
    assert row["Official_Close_Marker_Rows"] == 1
    assert row["Post_Market_Val"] == 0.0
    assert row["Total_Daily_Val"] == row["Open_Auction_Val"] + row["Close_Auction_Val"] + 70.0


def _volume_row(ticker: str = "AAPL") -> dict:
    return {
        "Ticker": ticker,
        "Date": dt.date(2018, 1, 2),
        "Is_Witching_Day": False,
        "Pre_Market_Val": 1.0,
        "Open_Auction_Val": 2.0,
        "Morning_30m_Val": 3.0,
        "Mid_Day_Val": 4.0,
        "Afternoon_30m_Val": 5.0,
        "Close_Auction_Val": 6.0,
        "Post_Market_Val": 7.0,
        "Official_Close_Marker_Val": 8.0,
        "Official_Close_Marker_Rows": 1,
        "Total_Daily_Val": 28.0,
    }


def test_volume_db_upsert_skips_are_replaced_by_successful_row() -> None:
    con = volume_db.init_db(":memory:")

    skip = {
        "Ticker": "AAPL",
        "Date": dt.date(2018, 1, 2),
        "Reason": "below_min_rows",
        "Detail": "only 3 rows",
        "Source_Path": "AAPL.parquet",
    }
    assert volume_db.upsert_skips(con, [skip]) == 1
    assert con.execute("select count(*) from daily_volume_skipped").fetchone()[0] == 1

    assert volume_db.upsert_batch(con, [_volume_row("AAPL")]) == 1
    assert con.execute("select count(*) from daily_volume_skipped").fetchone()[0] == 0
    row = con.execute(
        "select Ticker, Total_Daily_Val, Official_Close_Marker_Val "
        "from daily_volume where Ticker='AAPL'"
    ).fetchone()

    assert row == ("AAPL", 28.0, 8.0)
    con.close()


def test_volume_db_upsert_updates_existing_skip_detail() -> None:
    con = volume_db.init_db(":memory:")
    base = {
        "Ticker": "MSFT",
        "Date": dt.date(2018, 1, 2),
        "Reason": "compute_error",
        "Detail": "first",
        "Source_Path": "MSFT.parquet",
    }
    updated = {**base, "Detail": "second"}

    volume_db.upsert_skips(con, [base])
    volume_db.upsert_skips(con, [updated])

    rows = con.execute(
        "select count(*), min(Detail) from daily_volume_skipped where Ticker='MSFT'"
    ).fetchone()
    assert rows == (1, "second")
    con.close()
