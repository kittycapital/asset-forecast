#!/usr/bin/env python3
"""기존 CSV 파일의 중복 날짜 정리 (1회 실행용)"""
import pandas as pd
import glob
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

for filepath in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
    name = os.path.basename(filepath)
    df = pd.read_csv(filepath)
    before = len(df)
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values("Date").reset_index(drop=True)
    after = len(df)
    removed = before - after
    if removed > 0:
        df.to_csv(filepath, index=False)
        print(f"✓ {name}: {removed}개 중복 제거 ({before} → {after})")
    else:
        print(f"  {name}: 중복 없음")
