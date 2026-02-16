#!/usr/bin/env python3
"""
Asset Price Forecast Generator
Generates daily/weekly/monthly forecasts with confidence intervals.
Uses Exponential Smoothing + Monte Carlo simulation for prediction bands.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import json
import os
import glob
from datetime import datetime, timedelta

# ============================================================
# Configuration
# ============================================================

ASSET_META = {
    "SPY":     {"name": "S&P 500 ETF",        "category": "us_equity", "color": "#4fc3f7"},
    "QQQ":     {"name": "Nasdaq 100 ETF",      "category": "us_equity", "color": "#ab47bc"},
    "DIA":     {"name": "Dow Jones ETF",        "category": "us_equity", "color": "#42a5f5"},
    "IWM":     {"name": "Russell 2000 ETF",     "category": "us_equity", "color": "#66bb6a"},
    "ARKK":    {"name": "ARK Innovation ETF",   "category": "us_equity", "color": "#ff7043"},
    "EWY":     {"name": "South Korea ETF",      "category": "global",    "color": "#26c6da"},
    "EEM":     {"name": "Emerging Markets ETF",  "category": "global",    "color": "#8d6e63"},
    "SCHD":    {"name": "Dividend ETF",         "category": "dividend",  "color": "#5c6bc0"},
    "GLD":     {"name": "Gold ETF",             "category": "commodity", "color": "#ffd700"},
    "SLV":     {"name": "Silver ETF",           "category": "commodity", "color": "#b0bec5"},
    "USO":     {"name": "Oil ETF",              "category": "commodity", "color": "#a1887f"},
    "XLE":     {"name": "Energy Sector ETF",    "category": "commodity", "color": "#ef5350"},
    "TLT":     {"name": "20+ Year Treasury ETF","category": "bond",      "color": "#78909c"},
    "BTC_USD": {"name": "Bitcoin",              "category": "crypto",    "color": "#f7931a"},
    "ETH_USD": {"name": "Ethereum",             "category": "crypto",    "color": "#627eea"},
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

FORECAST_HORIZON = {
    "daily":  90,   # 90일 예측
    "weekly": 26,   # 26주 예측
    "monthly": 12,  # 12개월 예측
}

BACKTEST_WINDOW = 90  # 백테스트 기간 (일)


# ============================================================
# Data Loading
# ============================================================

def load_asset_data(filepath):
    """CSV 로드 및 전처리"""
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[["Date", "Close", "High", "Low", "Open", "Volume"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df


def resample_weekly(df):
    """주간 데이터로 리샘플링"""
    df = df.set_index("Date")
    weekly = df["Close"].resample("W-FRI").last().dropna()
    return weekly.reset_index()


def resample_monthly(df):
    """월간 데이터로 리샘플링"""
    df = df.set_index("Date")
    monthly = df["Close"].resample("ME").last().dropna()
    return monthly.reset_index()


# ============================================================
# Forecasting Model
# ============================================================

def compute_log_returns(prices):
    """로그 수익률 계산"""
    return np.diff(np.log(prices))


def forecast_monte_carlo(prices, n_days, n_simulations=5000):
    """
    Monte Carlo 시뮬레이션 기반 예측
    - 최근 데이터에 가중치를 둔 변동성 추정
    - GBM (Geometric Brownian Motion) 모델
    """
    log_returns = compute_log_returns(prices)

    # 최근 60일 수익률에 가중 (지수가중)
    recent_window = min(60, len(log_returns))
    recent_returns = log_returns[-recent_window:]
    weights = np.exp(np.linspace(-1, 0, len(recent_returns)))
    weights /= weights.sum()

    mu = np.average(recent_returns, weights=weights)
    sigma = np.sqrt(np.average((recent_returns - mu) ** 2, weights=weights))

    # 추세 조정: 최근 30일 vs 전체 추세 블렌딩
    recent_30 = log_returns[-min(30, len(log_returns)):]
    trend_short = np.mean(recent_30)
    trend_long = np.mean(log_returns[-min(252, len(log_returns)):])
    mu_adjusted = 0.6 * trend_short + 0.4 * trend_long

    last_price = prices[-1]
    simulations = np.zeros((n_simulations, n_days))

    for i in range(n_simulations):
        daily_returns = np.random.normal(mu_adjusted, sigma, n_days)
        price_path = last_price * np.exp(np.cumsum(daily_returns))
        simulations[i] = price_path

    # 백분위수 계산
    predicted = np.median(simulations, axis=0)
    lower_90 = np.percentile(simulations, 5, axis=0)
    upper_90 = np.percentile(simulations, 95, axis=0)
    lower_70 = np.percentile(simulations, 15, axis=0)
    upper_70 = np.percentile(simulations, 85, axis=0)

    # 대표 시뮬레이션 경로 20개 샘플링 (7일 시뮬레이션 뷰용)
    sample_indices = np.random.choice(n_simulations, size=min(20, n_simulations), replace=False)
    sample_paths = simulations[sample_indices]

    return {
        "predicted": predicted,
        "lower_70": lower_70,
        "upper_70": upper_70,
        "lower_90": lower_90,
        "upper_90": upper_90,
        "sample_paths": sample_paths,
    }


def generate_forecast(df, period="daily"):
    """특정 주기의 예측 생성"""
    if period == "daily":
        prices = df["Close"].values
        last_date = df["Date"].iloc[-1]
        n_steps = FORECAST_HORIZON["daily"]
        dates = [last_date + timedelta(days=i + 1) for i in range(n_steps)]
    elif period == "weekly":
        weekly = resample_weekly(df)
        prices = weekly["Close"].values
        last_date = weekly["Date"].iloc[-1]
        n_steps = FORECAST_HORIZON["weekly"]
        dates = [last_date + timedelta(weeks=i + 1) for i in range(n_steps)]
    elif period == "monthly":
        monthly = resample_monthly(df)
        prices = monthly["Close"].values
        last_date = monthly["Date"].iloc[-1]
        n_steps = FORECAST_HORIZON["monthly"]
        dates = [last_date + pd.DateOffset(months=i + 1) for i in range(n_steps)]

    result = forecast_monte_carlo(prices, n_steps)

    forecast_data = []
    for i in range(n_steps):
        entry = {
            "date": dates[i].strftime("%Y-%m-%d"),
            "predicted": round(float(result["predicted"][i]), 2),
            "lower_70": round(float(result["lower_70"][i]), 2),
            "upper_70": round(float(result["upper_70"][i]), 2),
            "lower_90": round(float(result["lower_90"][i]), 2),
            "upper_90": round(float(result["upper_90"][i]), 2),
        }
        forecast_data.append(entry)

    return forecast_data


def generate_simulation_7d(df):
    """7일 시뮬레이션 생성: 경로 20개 + 밴드 + 중앙 예측"""
    prices = df["Close"].values
    last_date = df["Date"].iloc[-1]
    last_price = float(prices[-1])
    n_days = 7
    dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days + 1)]

    result = forecast_monte_carlo(prices, n_days)

    # 시작점 (현재가) 포함한 경로 데이터
    paths = []
    for path in result["sample_paths"]:
        p = [round(last_price, 2)] + [round(float(v), 2) for v in path]
        paths.append(p)

    sim_data = {
        "dates": dates,
        "start_price": round(last_price, 2),
        "predicted": [round(last_price, 2)] + [round(float(v), 2) for v in result["predicted"]],
        "lower_70": [round(last_price, 2)] + [round(float(v), 2) for v in result["lower_70"]],
        "upper_70": [round(last_price, 2)] + [round(float(v), 2) for v in result["upper_70"]],
        "lower_90": [round(last_price, 2)] + [round(float(v), 2) for v in result["lower_90"]],
        "upper_90": [round(last_price, 2)] + [round(float(v), 2) for v in result["upper_90"]],
        "paths": paths,
    }

    return sim_data


# ============================================================
# Backtesting
# ============================================================

def backtest(df, period="daily"):
    """백테스트: 과거 데이터를 잘라서 예측 정확도 측정"""
    if period == "daily":
        prices = df["Close"].values
        test_window = min(BACKTEST_WINDOW, len(prices) // 4)
        horizons = [1, 7, 30]
    elif period == "weekly":
        weekly = resample_weekly(df)
        prices = weekly["Close"].values
        test_window = min(26, len(prices) // 4)
        horizons = [1, 4, 13]
    elif period == "monthly":
        monthly = resample_monthly(df)
        prices = monthly["Close"].values
        test_window = min(12, len(prices) // 4)
        horizons = [1, 3, 6]

    errors = {h: [] for h in horizons}

    for step in range(test_window, 0, -1):
        train_prices = prices[: -step]
        if len(train_prices) < 60:
            continue

        for h in horizons:
            if step - h < 0:
                continue
            actual_idx = len(prices) - step + h - 1
            if actual_idx >= len(prices):
                continue

            result = forecast_monte_carlo(train_prices, h, n_simulations=500)
            predicted = result["predicted"][-1]
            actual = prices[actual_idx]

            if actual > 0:
                ape = abs(predicted - actual) / actual * 100
                errors[h].append(ape)

    mape_results = {}
    for h in horizons:
        if errors[h]:
            mape_results[h] = round(float(np.mean(errors[h])), 2)
        else:
            mape_results[h] = None

    return mape_results


# ============================================================
# Main Pipeline
# ============================================================

def process_asset(asset_key, filepath):
    """단일 자산 처리: 예측 + 백테스트 → JSON 출력"""
    print(f"  Processing {asset_key}...")
    df = load_asset_data(filepath)
    meta = ASSET_META[asset_key]

    # 최근 1년 히스토리컬 데이터 (차트 표시용)
    one_year_ago = df["Date"].iloc[-1] - timedelta(days=365)
    historical = df[df["Date"] >= one_year_ago][["Date", "Close"]].copy()
    historical_data = [
        {"date": row["Date"].strftime("%Y-%m-%d"), "price": round(float(row["Close"]), 2)}
        for _, row in historical.iterrows()
    ]

    # 예측 생성
    print(f"    → Daily forecast...")
    daily_forecast = generate_forecast(df, "daily")
    print(f"    → Weekly forecast...")
    weekly_forecast = generate_forecast(df, "weekly")
    print(f"    → Monthly forecast...")
    monthly_forecast = generate_forecast(df, "monthly")
    print(f"    → 7-day simulation...")
    sim_7d = generate_simulation_7d(df)

    # 백테스트
    print(f"    → Backtesting...")
    daily_bt = backtest(df, "daily")
    weekly_bt = backtest(df, "weekly")
    monthly_bt = backtest(df, "monthly")

    # JSON 구조
    output = {
        "asset": asset_key,
        "name": meta["name"],
        "category": meta["category"],
        "color": meta["color"],
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_price": round(float(df["Close"].iloc[-1]), 2),
        "last_date": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "historical": historical_data,
        "forecast": {
            "daily": daily_forecast,
            "weekly": weekly_forecast,
            "monthly": monthly_forecast,
        },
        "simulation_7d": sim_7d,
        "backtest": {
            "daily": {
                "mape_1d": daily_bt.get(1),
                "mape_7d": daily_bt.get(7),
                "mape_30d": daily_bt.get(30),
            },
            "weekly": {
                "mape_1w": weekly_bt.get(1),
                "mape_4w": weekly_bt.get(4),
                "mape_13w": weekly_bt.get(13),
            },
            "monthly": {
                "mape_1m": monthly_bt.get(1),
                "mape_3m": monthly_bt.get(3),
                "mape_6m": monthly_bt.get(6),
            },
        },
    }

    return output


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_forecasts = {}
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    print(f"Found {len(csv_files)} CSV files in {DATA_DIR}")
    print("=" * 50)

    for filepath in sorted(csv_files):
        asset_key = os.path.basename(filepath).replace(".csv", "")
        if asset_key not in ASSET_META:
            print(f"  Skipping unknown asset: {asset_key}")
            continue

        result = process_asset(asset_key, filepath)
        all_forecasts[asset_key] = result

        # 개별 JSON 저장
        out_path = os.path.join(OUTPUT_DIR, f"{asset_key}_forecast.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    ✓ Saved {out_path}")

    # 통합 JSON (프론트엔드용)
    combined_path = os.path.join(OUTPUT_DIR, "all_forecasts.json")
    with open(combined_path, "w") as f:
        json.dump(all_forecasts, f, indent=2)
    print(f"\n✓ Combined JSON saved: {combined_path}")
    print(f"✓ Processed {len(all_forecasts)} assets total")


if __name__ == "__main__":
    main()
