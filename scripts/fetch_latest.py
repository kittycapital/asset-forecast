#!/usr/bin/env python3
"""
Fetch latest price data for all assets.
- Yahoo Finance (yfinance) for stocks/ETFs
- CoinGecko API for crypto

Appends new rows to existing CSV files in data/raw/
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_last_date(filepath):
    """CSV의 마지막 날짜 확인"""
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, parse_dates=["Date"])
    return df["Date"].max()


# ============================================================
# Yahoo Finance Fetcher
# ============================================================
def fetch_yahoo(ticker, start_date):
    """yfinance로 최신 데이터 가져오기"""
    import yfinance as yf

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_str = (start_date + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"  Fetching {ticker} from {start_str} to {end_date}...")
    data = yf.download(ticker, start=start_str, end=end_date, progress=False)

    if data.empty:
        print(f"  No new data for {ticker}")
        return None

    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data[["Close", "High", "Low", "Open", "Volume"]].reset_index()
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    print(f"  → Got {len(df)} new rows")
    return df


# ============================================================
# CoinGecko Fetcher
# ============================================================
def fetch_coingecko(coingecko_id, start_date):
    """CoinGecko API로 암호화폐 데이터 가져오기"""
    import requests

    days_diff = (datetime.now() - start_date).days
    if days_diff <= 0:
        print(f"  No new data for {coingecko_id}")
        return None

    # market_chart API 사용 (OHLC보다 유연한 days 값 허용)
    url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"
    params = {"vs_currency": "usd", "days": min(days_diff + 1, 90)}

    print(f"  Fetching {coingecko_id} last {params['days']} days...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data or "prices" not in data:
        return None

    rows = []
    seen_dates = set()
    for entry in data["prices"]:
        ts, price = entry
        dt = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        if dt not in seen_dates:
            seen_dates.add(dt)
            rows.append({"Date": dt, "Close": price, "High": price, "Low": price, "Open": price, "Volume": 0})

    df = pd.DataFrame(rows)
    # Filter only after start_date
    df = df[df["Date"] > start_date.strftime("%Y-%m-%d")]

    print(f"  → Got {len(df)} new rows")
    return df if len(df) > 0 else None


# ============================================================
# Main
# ============================================================
def main():
    config = load_config()
    assets = config["assets"]

    print("=" * 50)
    print(f"Fetching latest data — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    updated = 0

    for key, meta in assets.items():
        filepath = os.path.join(RAW_DIR, f"{key}.csv")
        last_date = get_last_date(filepath)

        if last_date is None:
            print(f"\n⚠ {key}: No CSV found, skipping")
            continue

        print(f"\n{key} — last data: {last_date.strftime('%Y-%m-%d')}")

        try:
            if meta["source"] == "yahoo":
                new_data = fetch_yahoo(meta["ticker"], last_date)
            elif meta["source"] == "coingecko":
                new_data = fetch_coingecko(meta["coingecko_id"], last_date)
            else:
                print(f"  Unknown source: {meta['source']}")
                continue

            if new_data is not None and len(new_data) > 0:
                # Append to existing CSV, deduplicate by date (keep last)
                existing = pd.read_csv(filepath)
                combined = pd.concat([existing, new_data], ignore_index=True)
                combined = combined.drop_duplicates(subset=["Date"], keep="last")
                combined = combined.sort_values("Date").reset_index(drop=True)
                combined.to_csv(filepath, index=False)
                print(f"  ✓ Updated {filepath}")
                updated += 1
            else:
                print(f"  — Already up to date")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n{'=' * 50}")
    print(f"Done. Updated {updated}/{len(assets)} assets.")
    return 0 if updated >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
