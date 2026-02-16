# 📊 Asset Forecast

15개 글로벌 자산의 가격 예측 대시보드. Monte Carlo GBM 모델 기반 Daily/Weekly/Monthly 예측 + 백테스트 정확도 제공.

## 자산 목록

| Category | Assets |
|----------|--------|
| US Stocks | SPY, QQQ, DIA, IWM, ARKK |
| Global | EWY, EEM |
| Dividend | SCHD |
| Commodities | GLD, SLV, USO, XLE |
| Bonds | TLT |
| Crypto | BTC, ETH |

## 구조

```
asset-forecast/
├── data/raw/           # 15 CSV 파일 (원본 데이터)
├── scripts/
│   ├── fetch_latest.py # API 데이터 수집
│   └── forecast.py     # 예측 모델 + 백테스트
├── output/             # 생성된 JSON
├── docs/               # GitHub Pages (프론트엔드)
│   ├── index.html
│   └── data/
└── .github/workflows/  # 자동 업데이트
```

## 로컬 실행

```bash
pip install -r requirements.txt

# 예측 실행
python scripts/forecast.py

# JSON을 docs로 복사
cp output/all_forecasts.json docs/data/

# 로컬 서버 실행
cd docs && python -m http.server 8000
```

## GitHub Pages 배포

1. Repository Settings → Pages → Source: `Deploy from a branch`
2. Branch: `main`, Folder: `/docs`
3. Save → 자동 배포됨

## 자동 업데이트

GitHub Actions가 매일 한국시간 15:00 (미장 마감 후)에 자동 실행:
1. 최신 가격 데이터 수집
2. 예측 모델 재학습
3. JSON 업데이트 및 커밋

수동 실행: Actions 탭 → "Daily Forecast Update" → Run workflow

## 모델 설명

- **Monte Carlo GBM** (Geometric Brownian Motion)
- 최근 데이터에 지수가중치를 둔 변동성 추정
- 5,000회 시뮬레이션으로 예측 분포 생성
- 70% / 90% 신뢰구간 밴드 제공
- MAPE (평균 절대 백분율 오차) 기반 백테스트
