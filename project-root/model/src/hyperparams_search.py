#!/usr/bin/env python3
import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error
from model import Seq2Seq, device
from dataset import TimeSeriesDataset

def random_search_hyperparams(
    df, feature_cols, n_trials=10, in_len=1440, out_len=4
):
    """
    df: pandas DataFrame with all features + 'Close'
    feature_cols: list of column names to use as X
    """
    best_mae = float('inf')
    best_params = None
    best_scalers = (None, None)

    # 1) 전체 데이터셋 윈도우 생성
    ds = TimeSeriesDataset(df, feature_cols, in_len, out_len)
    scaler_X, scaler_y = ds.get_scalers()
    n = len(ds)
    train_n = int(n * 0.7)
    val_n   = int(n * 0.15)

    for trial in range(1, n_trials + 1):
        # 2) 하이퍼파라미터 샘플링
        hid_dim    = random.choice([32, 64, 128, 256])
        n_layers   = random.choice([1, 2, 3])
        lr         = random.choice([1e-4, 3e-4, 1e-3, 3e-3])
        batch_size = random.choice([16, 32, 64])

        # 3) DataLoader 분할
        train_loader = DataLoader(
            Subset(ds, range(0, train_n)),
            batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            Subset(ds, range(train_n, train_n + val_n)),
            batch_size=batch_size, shuffle=False
        )

        # 4) 모델 초기화
        model = Seq2Seq(len(feature_cols), 1, hid_dim, out_len, n_layers).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # 5) 간단 학습 (5 에포크)
        for _ in range(5):
            model.train()
            for xb, yb, y0b in train_loader:
                xb, yb, y0b = xb.to(device), yb.to(device), y0b.to(device)
                loss = loss_fn(model(xb, y0b), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # 6) 검증 MAE 계산
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, yb, y0b in val_loader:
                xb, yb, y0b = xb.to(device), yb.to(device), y0b.to(device)
                p = model(xb, y0b).squeeze(-1).cpu().numpy()
                t = yb.squeeze(-1).cpu().numpy()
                all_preds.append(p.flatten())
                all_trues.append(t.flatten())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_trues)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"Trial {trial}/{n_trials}: hid={hid_dim}, layers={n_layers}, "
              f"lr={lr}, bs={batch_size} → Val MAE: {mae:.4f}")

        # 7) 최적 갱신
        if mae < best_mae:
            best_mae = mae
            best_params = {
                'hid_dim': hid_dim,
                'n_layers': n_layers,
                'learning_rate': lr,
                'batch_size': batch_size
            }
            best_scalers = (scaler_X, scaler_y)

    print("\n=== Best Hyperparams ===")
    print(best_params, f"MAE={best_mae:.4f}")

    return best_params, best_scalers[0], best_scalers[1]


if __name__ == "__main__":
    # ─── 랜덤 서치 실행을 위한 데이터 로드
    import pandas as pd
    from loader import load_candles_via_db

    df = load_candles_via_db()
    feature_cols = [
        'Open','High','Low','Close','Volume','total_transaction',
        'SMA','Upper_Band','Lower_Band','MACD','MACD_Signal',
        'RSI','EMA_50','EMA_200'
    ]

    # ─── 랜덤 서치 수행
    best_params, scaler_X, scaler_y = random_search_hyperparams(
        df, feature_cols,
        n_trials=10,    # 필요에 따라 조정
        in_len=1440,    # 예: 15분 * 1440 = 15일치
        out_len=4       # 4스텝 예측
    )

    # ─── 결과를 model/config/best_params.json 에 저장
    #    (호스트의 ./model/config 과 매핑되는 위치)
    script_dir    = os.path.abspath(os.path.dirname(__file__))       # .../model/src
    model_root    = os.path.abspath(os.path.join(script_dir, '..'))  # .../model
    config_dir    = os.path.join(model_root, 'config')
    os.makedirs(config_dir, exist_ok=True)

    config_path   = os.path.join(config_dir, 'best_params.json')
    with open(config_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"\n>> Saved best_params to {config_path}")
