import pandas as pd
from loader import load_candles_via_db
from src.utils import forecast_next_hour

def load_data():
    """
    DB에서 캔들+지표를 불러와서 pandas.DataFrame 으로 반환.
    인덱스는 datetime 타입이어야 plot 時 x축에 날짜가 표시됩니다.
    """
    df = load_candles_via_db()
    # 예: df.index = pd.to_datetime(df["timestamp"])
    return df
