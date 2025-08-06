#!/usr/bin/env python3
import os
import sys
import time
import datetime
import json

import torch
import joblib
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 프로젝트 루트를 PYTHONPATH에 추가 (train.py 위치: /app/src/forecast_scheduler.py → ROOT=/app)
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(__file__, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# DB에서 캔들 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────
from loader import load_candles_via_db

# ─────────────────────────────────────────────────────────────────────────────
# 예측 유틸 함수
# ─────────────────────────────────────────────────────────────────────────────
from utils import forecast_next_hour

# ─────────────────────────────────────────────────────────────────────────────
# Seq2Seq 모델 클래스, device
# ─────────────────────────────────────────────────────────────────────────────
from model import Seq2Seq, device

# ─────────────────────────────────────────────────────────────────────────────
# 설정: feature 컬럼, 윈도우 크기, 예측 스텝
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'Open','High','Low','Volume','total_transaction',
    'SMA','Upper_Band','Lower_Band','MACD','MACD_Signal',
    'RSI','EMA_50','EMA_200'
]
IN_LEN  = 1440  # 과거 2,880스텝(15분 * 2,880 = 30,000분 ≒ 500시간) ← 필요에 맞게 조정
OUT_LEN = 4     # 한 번에 4스텝(1시간) 예측

# ─────────────────────────────────────────────────────────────────────────────
# 모델 가중치·스케일러 로드
# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts():
    # best_params.json 에 저장된 파라미터 읽기
    cfg_path = os.path.join(ROOT, 'model', 'config', 'best_params.json')
    params = json.load(open(cfg_path, 'r'))

    # Seq2Seq 인스턴스 생성 (train.py 와 동일하게 매핑)
    remap = {
        'lstm_layers': int(params['n_layers']),
        'dense_layers': 0,
        'units': int(params['hid_dim']),
        'dense_units': int(params['hid_dim']),
        'learning_rate': float(params['learning_rate']),
        'lstm_dropout_rate': 0.0,
        'dense_dropout_rate': 0.0,
        'attention_dropout_rate': 0.0,
        'batch_size': int(params['batch_size'])
    }

    model = Seq2Seq(
        enc_input_dim=len(FEATURE_COLS),
        dec_input_dim=1,
        hid_dim=remap['units'],
        out_len=OUT_LEN,
        n_layers=remap['lstm_layers'],
        dropout=remap['lstm_dropout_rate']
    ).to(device)
    # 저장된 모델 가중치 로드
    weight_path = os.environ.get(
        'OUTPUT_DIR',
        os.path.join(ROOT, 'model', 'output')
    )
    model.load_state_dict(torch.load(
        os.path.join(weight_path, 'model.pth'),
        map_location=device
    ))
    model.eval()

    # 스케일러 로드
    scaler_path = os.path.join(weight_path, 'scaler.pkl')
    scalers = joblib.load(scaler_path)
    return model, scalers['scaler_X'], scalers['scaler_y']

# ─────────────────────────────────────────────────────────────────────────────
# 매 정각 예측 및 출력
# ─────────────────────────────────────────────────────────────────────────────
def run_hourly_forecast(model, scaler_X, scaler_y):
    # 1) 최신 캔들 데이터 로드
    df = load_candles_via_db()

    # 1a) 만약 df.index가 datetime이 아니면, timestamp 컬럼을 인덱스로 설정
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

    # 2) 예측
    future = forecast_next_hour(
        model, scaler_X, scaler_y,
        df, FEATURE_COLS,
        in_len=IN_LEN,
        device=device
    )

    # 3) 시간 레이블
    now          = datetime.datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    prev_hour    = current_hour - datetime.timedelta(hours=1)

    # e.g. "7.7 13시"
    fmt = lambda dt: f"{dt.month}.{dt.day} {dt.hour}시"

    # 4) 예측값 출력
    pred_str = ", ".join(f"{v:.2f}" for v in future)
    print(f"{fmt(current_hour)} 예측값 : {pred_str}")

    # 5) 실제값 출력 (직전 정각부터 15분 간격 4포인트)
    actuals = []
    for i in range(1, OUT_LEN+1):
        ts = prev_hour + datetime.timedelta(minutes=15 * i)
        # 정확한 timestamp가 없으면 asof로 가장 가까운 과거 값 사용
        try:
            val = df.at[ts, 'Close']
        except KeyError:
            val = df['Close'].asof(ts)
        actuals.append(val)

    act_str = ", ".join(f"{v:.2f}" for v in actuals)
    print(f"{fmt(prev_hour)} 실제값 : {act_str}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 메인 스케줄러: 매 정각 대기 후 예측 실행
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model, scaler_X, scaler_y = load_artifacts()

    while True:
        now = datetime.datetime.now()
        # 다음 정각 시각 계산
        next_run = (now + datetime.timedelta(hours=1)) \
                       .replace(minute=0, second=0, microsecond=0)
        wait_sec = (next_run - now).total_seconds()
        print(f"다음 예측까지 대기: {int(wait_sec)}초 (실행 시각: {next_run.strftime('%Y-%m-%d %H:%M')})")
        time.sleep(wait_sec)

        run_hourly_forecast(model, scaler_X, scaler_y)

