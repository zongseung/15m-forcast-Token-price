#!/usr/bin/env python3
# coding: utf-8

import time
import pandas as pd
from preprocess import candle_chart, compute_bollinger_bands_timebased, compute_technical_indicators
from db_client import DBClient

def backfill(interval="15m", batch_size=1000):
    """
    interval: 15m, 1h 등
    batch_size: 한 번에 가져올 캔들 개수 (최대 API 허용량에 따라 조정)
    """
    client = DBClient()
    now_ms = int(pd.Timestamp.now().timestamp() * 1000)
    end_time = now_ms
    total_upserted = 0

    while True:
        # startTime 을 batch_size 개 전으로 맞춰 계산
        df_batch = candle_chart(interval, 
                                start_time=0, 
                                end_time=end_time)
        if df_batch is None or df_batch.empty:
            break

        # 페이징: 가장 오래된 타임스탬프에서 batch_size 만큼만
        df_batch = df_batch.sort_index(ascending=False).head(batch_size)
        start_time = int(df_batch.index.min().timestamp() * 1000)

        # 지표 계산
        bb = compute_bollinger_bands_timebased(df_batch["Close"], interval, span_hours=20)
        tech = compute_technical_indicators(df_batch, interval)
        df_full = pd.concat([df_batch, bb, tech], axis=1).dropna()

        # DB에 업서트
        n = client.upsert_candles(df_full)
        total_upserted += n
        print(f"▶️ Upserted {n} rows for period ending {pd.to_datetime(end_time, unit='ms')}")

        # 다음 루프에서 이 batch 바로 이전 시점까지 가져오도록
        end_time = start_time - 1

        # 더 가져올 데이터가 있는지 확인
        if end_time <= 0:
            break

        # 너무 빠르게 호출하지 않도록 약간 대기
        time.sleep(0.5)

    print(f"✅ Backfill complete, total upserted: {total_upserted}")

if __name__ == "__main__":
    backfill(interval="15m", batch_size=1000)
