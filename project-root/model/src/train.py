#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

# ───────────────────────────────────────────────────────────────────────────────
# PROJECT ROOT
ROOT = os.path.abspath(os.path.join(__file__, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ───────────────────────────────────────────────────────────────────────────────
# OUTPUT DIR
DEFAULT_OUT = os.path.join(ROOT, 'model', 'output')
OUTPUT_DIR  = os.environ.get('OUTPUT_DIR', DEFAULT_OUT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────────
# DATA + DATASET
from loader import load_candles_via_db
from dataset import TimeSeriesDataset

# ───────────────────────────────────────────────────────────────────────────────
# MODEL + FORECAST UTIL
from model import train_model, device
from utils import forecast_next_hour

# ───────────────────────────────────────────────────────────────────────────────
# RANDOM SEARCH
from hyperparams_search import random_search_hyperparams

# enable cuDNN benchmark
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


def get_best_params(config_path, df, feature_cols, force_update=False):
    if not force_update and os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            bp = json.load(f)
        print("Loaded hyperparameters:", bp)
        return bp, None, None

    print("Running random search to update best hyperparameters…")
    bp, sX, sY = random_search_hyperparams(
        df, feature_cols,
        n_trials=10,
        in_len=1440,
        out_len=4
    )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(bp, f, indent=2)
    print("Saved hyperparameters:", bp)
    return bp, sX, sY



def main():
    # 1) load
    print("Loading data from DB…")
    df = load_candles_via_db()
    feature_cols = [
        'Open','High','Low','Volume','total_transaction',
        'SMA','Upper_Band','Lower_Band',
        'MACD','MACD_Signal','RSI','EMA_50','EMA_200'
    ]

    # 2) hyperparams
    cfg = os.path.join(ROOT, 'model', 'config', 'best_params.json')
    best_params, scaler_X, scaler_y = get_best_params(cfg, df, feature_cols, force_update=True)


    # 3) lengths
    in_len  = best_params.get('in_len', 1440)
    out_len = best_params.get('out_len', 4)

    # 4) dataset
    ds = TimeSeriesDataset(df, feature_cols, in_len=in_len, out_len=out_len)
    if scaler_X is None or scaler_y is None:
        scaler_X, scaler_y = ds.get_scalers()

    # 5) split
    n    = len(ds)
    i_tr = int(n*0.7)
    i_va = int(n*0.85)

    def collate(idxs):
        Xs, Ys = [], []
        for i in idxs:
            x, y, _ = ds[i]
            Xs.append(x.numpy()); Ys.append(y.numpy().flatten())
        return np.stack(Xs), np.stack(Ys)

    X_tr, Y_tr = collate(range(0, i_tr))
    X_va, Y_va = collate(range(i_tr, i_va))

    # non‐overlapping test
    test_idxs = list(range(i_va, n, out_len))
    X_te, Y_te = collate(test_idxs)
    print(f"Test blocks: {len(test_idxs)} × {out_len} steps")

    # 6) map to train_model args
    remap = {
        'lstm_layers':       int(best_params['n_layers']),
        'dense_layers':      0,
        'units':             int(best_params['hid_dim']),
        'dense_units':       int(best_params['hid_dim']),
        'learning_rate':     float(best_params['learning_rate']),
        'lstm_dropout_rate': 0.0,
        'dense_dropout_rate':0.0,
        'attention_dropout_rate':0.0,
        'batch_size':        int(best_params['batch_size'])
    }

    # 7) train & test
    model, tr_loss, va_loss, te_loss, te_preds = train_model(
        X_tr, Y_tr,
        X_va, Y_va,
        X_te, Y_te,
        remap,
        device
    )
    print(f"Normalized test loss: {te_loss:.6f}")
    print(f"te_preds.shape = {te_preds.shape}")

    # 8) inverse scale
    real_pred = scaler_y.inverse_transform(te_preds)
    real_true = scaler_y.inverse_transform(Y_te)

    # 9) few samples
    print("Sample predictions:")
    for i in range(min(5, len(real_true))):
        print(f" block {i}: true={real_true[i].tolist()} → pred={real_pred[i].tolist()}")

    # 10) future forecast
    future = forecast_next_hour(
        model, scaler_X, scaler_y,
        df, feature_cols,
        in_len=in_len,
        device=device
    )
    print("Future 4-step forecast:")
    for t, v in enumerate(future, 1):
        print(f"  +{t*15}min => {v:.4f}")

    # 11) save artifacts
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pth'))
    joblib.dump({'scaler_X':scaler_X,'scaler_y':scaler_y},
                os.path.join(OUTPUT_DIR,'scaler.pkl'))
    pd.DataFrame({
        'actual': real_true.flatten(),
        'predicted': real_pred.flatten()
    }).to_csv(os.path.join(OUTPUT_DIR,'test_predictions.csv'), index=False)

    # 12) global timeline plot
    r2   = r2_score(real_true.flatten(), real_pred.flatten())
    mape = mean_absolute_percentage_error(real_true.flatten(), real_pred.flatten())
    rmse = np.sqrt(mean_squared_error(real_true.flatten(), real_pred.flatten()))

    # build global prediction series
    full_len    = len(df)
    preds_global = np.full(full_len, np.nan)
    for b_idx, ds_idx in enumerate(test_idxs):
        start = ds_idx + in_len
        for k in range(out_len):
            pos = start + k
            if pos < full_len:
                preds_global[pos] = real_pred[b_idx, k]

    plt.figure(figsize=(12,5))
    plt.plot(df['Close'].values, label='Actual Close')
    plt.plot(preds_global, '--', label='Forecast', alpha=0.8)
    plt.title(f"R2={r2:.3f}, MAPE={mape:.3f}, RMSE={rmse:.3f}")
    plt.xlabel("Time Index")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_timeline.png'))
    plt.close()

    # 13) **TEST SEGMENT PLOT**
    #    non-overlapping test blocks 만 모아 실제 vs 예측 플롯
    # global timestamp index for each test‐step
    test_positions = []
    for b_idx, ds_idx in enumerate(test_idxs):
        base = ds_idx + in_len
        test_positions += [base + k for k in range(out_len)]
    # convert to actual timestamps
    test_dates = df.index[test_positions]

    plt.figure(figsize=(12,5))
    plt.plot(test_dates, real_true.flatten(),  label='Actual (test)')
    plt.plot(test_dates, real_pred.flatten(), '--', label='Predicted (test)')
    plt.title("Actual vs Predicted on Test Segment")
    plt.xlabel("Timestamp")
    plt.ylabel("Close Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_segment_plot.png'))
    plt.close()

    print(f"✅ All outputs saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
