#!/usr/bin/env python3
# coding: utf-8

import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    DateTime, Float
)
from sqlalchemy.dialects.mysql import insert as mysql_insert

# 프로젝트 루트의 .env 파일을 로드
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path)

class DBClient:
    def __init__(self):
        user = os.getenv("DB_USER")
        pwd  = os.getenv("DB_PASS")
        host = os.getenv("DB_HOST", "db")
        port = os.getenv("DB_PORT", "3306")
        db   = os.getenv("DB_NAME")
        url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"

        # SQLAlchemy 엔진 생성
        self.engine = create_engine(url, echo=False)

        # 테이블 정의 및 생성
        self._prepare_table()

    def _prepare_table(self):
        meta = MetaData()
        self.candles = Table(
            "candles", meta,
            Column("t",                DateTime, primary_key=True),
            Column("Open",             Float),
            Column("High",             Float),
            Column("Low",              Float),
            Column("Close",            Float),
            Column("Volume",           Float),
            Column("total_transaction",Float),
            Column("SMA",              Float),
            Column("Upper_Band",       Float),
            Column("Lower_Band",       Float),
            Column("MACD",             Float),
            Column("MACD_Signal",      Float),
            Column("RSI",              Float),
            Column("EMA_50",           Float),
            Column("EMA_200",          Float),
            mysql_engine='InnoDB',
            mysql_charset='utf8mb4'
        )
        meta.create_all(self.engine)

    def upsert_candles(self, df: pd.DataFrame) -> int:
        # 인덱스를 칼럼으로 옮기고, 컬럼명 교정
        df = df.reset_index().rename(columns={
            "Upper Band":  "Upper_Band",
            "Lower Band":  "Lower_Band",
            "total_transaction": "total_transaction"
        })
        # pandas.Timestamp → Python datetime 으로 변환
        df['t'] = df['t'].dt.to_pydatetime()

        # dict 리스트로 변환
        records = df.to_dict(orient="records")

        # MySQL 전용 upsert 구문 생성
        stmt = mysql_insert(self.candles).values(records)
        update_cols = {
            col.name: stmt.inserted[col.name]
            for col in self.candles.columns
            if col.name != "t"
        }
        upsert = stmt.on_duplicate_key_update(**update_cols)

        # 실행 및 영향받은 행(rowcount) 반환
        with self.engine.begin() as conn:
            result = conn.execute(upsert)
            return result.rowcount

