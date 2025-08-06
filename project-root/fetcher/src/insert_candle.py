# db_client.py
import os
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    BigInteger, Float, insert
)
from sqlalchemy.dialects.mysql import insert as mysql_insert
import pandas as pd

class DBClient:
    def __init__(self):
        user = os.getenv("DB_USER")
        pwd  = os.getenv("DB_PASS")
        host = os.getenv("DB_HOST", "db")
        port = os.getenv("DB_PORT", "3306")
        db   = os.getenv("DB_NAME")
        url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
        self.engine = create_engine(url, echo=False)
        self.meta   = MetaData()
        # 테이블 정의 (타임스탬프를 PK 로)
        self.candles = Table(
            "candles", self.meta,
            Column("t", BigInteger, primary_key=True),   # ms 단위 타임스탬프
            Column("Open",  Float),
            Column("High",  Float),
            Column("Low",   Float),
            Column("Close", Float),
            Column("Volume",Float),
            Column("SMA",       Float),
            Column("Upper_Band",Float),
            Column("Lower_Band",Float),
            Column("MACD",      Float),
            Column("MACD_Signal",Float),
            Column("RSI",       Float),
            Column("EMA_50",    Float),
            Column("EMA_200",   Float),
            mysql_charset="utf8mb4"
        )
        self.meta.create_all(self.engine)

    def upsert_candles(self, df: pd.DataFrame) -> int:
        """
        df.index: pandas.DatetimeIndex
        df columns: Open, High, Low, Close, Volume,
                    SMA, Upper Band, Lower Band,
                    MACD, MACD_Signal, RSI, EMA_50, EMA_200
        """
        # 1) DataFrame -> list of dict
        records = []
        for ts, row in df.iterrows():
            records.append({
                "t": int(ts.value // 10**6),  # numpy timestamp -> ms
                "Open":  row["Open"],
                "High":  row["High"],
                "Low":   row["Low"],
                "Close": row["Close"],
                "Volume":row["Volume"],
                "SMA":        row["SMA"],
                "Upper_Band": row["Upper Band"],
                "Lower_Band": row["Lower Band"],
                "MACD":       row["MACD"],
                "MACD_Signal":row["MACD_Signal"],
                "RSI":        row["RSI"],
                "EMA_50":     row["EMA_50"],
                "EMA_200":    row["EMA_200"],
            })

        # 2) MySQL 전용 upsert 구문 생성
        stmt = mysql_insert(self.candles).values(records)
        update_cols = {c.name: stmt.inserted[c.name] for c in self.candles.columns
                       if c.name != "t"}
        upsert_stmt = stmt.on_duplicate_key_update(**update_cols)

        # 3) 실행
        with self.engine.begin() as conn:
            result = conn.execute(upsert_stmt)
            # result.rowcount 는 “영향받은 행 수” (insert+update 합)
            return result.rowcount
