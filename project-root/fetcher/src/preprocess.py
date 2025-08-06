#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import requests

#############################################
# 1) 데이터 로딩 & 지표 계산 함수
#############################################
def interval_to_minutes(interval: str) -> int:
    unit, value = interval[-1], int(interval[:-1])
    if unit == 'm': return value
    if unit == 'h': return value * 60
    if unit == 'd': return value * 1440
    raise ValueError(f"Unsupported interval: {interval}")

def get_period_window(interval: str, span_hours: float=None, span_minutes: float=None) -> int:
    iv = interval_to_minutes(interval)
    if span_hours is not None:
        periods = span_hours * 60 / iv
    elif span_minutes is not None:
        periods = span_minutes / iv
    else:
        raise ValueError("span_hours or span_minutes must be set")
    return max(1, int(periods))

def candle_chart(interval: str, start_time: int=0, end_time: int=None) -> pd.DataFrame | None:
    if end_time is None:
        end_time = int(pd.Timestamp.now().timestamp() * 1000)
    body = {
      "type": "candleSnapshot",
      "req": {
        "coin": "HYPE",
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time
      }
    }
    resp = requests.post("https://api.hyperliquid.xyz/info",
                         headers={"Content-Type": "application/json"},
                         json=body)
    if resp.status_code != 200:
        print("API error:", resp.status_code)
        return None
    df = pd.DataFrame(resp.json())
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('t', inplace=True)
    df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close",
                       "v":"Volume","n":"total_transaction"},
              inplace=True)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = df[c].astype(float)
    df.drop(columns=["T","s","i"], errors='ignore', inplace=True)
    return df

def compute_bollinger_bands_timebased(series: pd.Series, interval: str,
                                      span_hours: float, num_std: int=2) -> pd.DataFrame:
    window = get_period_window(interval, span_hours=span_hours)
    m = series.rolling(window=window).mean()
    s = series.rolling(window=window).std()
    return pd.DataFrame({
        'SMA': m,
        'Upper Band': m + num_std * s,
        'Lower Band': m - num_std * s
    })

def compute_technical_indicators(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    close = df['Close']
    # MACD
    ema_short = close.ewm(span=get_period_window(interval, span_hours=12), adjust=False).mean()
    ema_long  = close.ewm(span=get_period_window(interval, span_hours=26), adjust=False).mean()
    macd_line = ema_short - ema_long
    macd_sig  = macd_line.ewm(span=get_period_window(interval, span_hours=9), adjust=False).mean()
    # RSI
    rsi_p = get_period_window(interval, span_hours=14)
    delta = close.diff().fillna(0)
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs  = up.rolling(window=rsi_p).mean() / down.rolling(window=rsi_p).mean()
    rsi = 100 - (100 / (1 + rs))
    # EMA
    ema50  = close.ewm(span=get_period_window(interval, span_hours=50),  adjust=False).mean()
    ema200 = close.ewm(span=get_period_window(interval, span_hours=200), adjust=False).mean()
    return pd.DataFrame({
        'MACD': macd_line,
        'MACD_Signal': macd_sig,
        'RSI': rsi,
        'EMA_50': ema50,
        'EMA_200': ema200
    })

def get_candles_with_indicators(interval="15m", bb_span_hours=20, num_std=2) -> pd.DataFrame | None:
    df = candle_chart(interval)
    if df is None:
        return None
    bb_df   = compute_bollinger_bands_timebased(df['Close'], interval,
                                                 span_hours=bb_span_hours, num_std=num_std)
    tech_df = compute_technical_indicators(df, interval)
    return pd.concat([df, bb_df, tech_df], axis=1).dropna()