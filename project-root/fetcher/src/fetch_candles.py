# fetch_candles.py

from preprocess import get_candles_with_indicators
from db_client import DBClient
import os
import time
import pandas as pd

def main():
    client = DBClient()
    # 1) API로 15분 단위 캔들+지표 DataFrame 가져오기
    df = get_candles_with_indicators(interval="15m", bb_span_hours=20, num_std=2)
    if df is None or df.empty:
        print("▶️ No data fetched.")
        return

    # 2) DB에 업서트
    n = client.upsert_candles(df)
    print(f"▶️ Upserted {n} rows at {pd.Timestamp.now()}")

if __name__ == "__main__":
    main()
