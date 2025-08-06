# model/src/loader.py

import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

def load_candles_via_db() -> pd.DataFrame:
    """
    .env에서 DB 접속 정보를 읽어와 MySQL 'candles' 테이블을 pandas DataFrame으로 반환합니다.

    필요 환경변수:
      - DB_HOST
      - DB_PORT
      - DB_NAME
      - DB_USER
      - DB_PASS

    예) .env 파일에 다음과 같이 설정
        DB_HOST=db
        DB_PORT=3306
        DB_NAME=hype_db
        DB_USER=hype_user
        DB_PASS=1234
    """
    # 1) .env 파일 로드
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=env_path)

    # 2) 환경변수 읽기
    db_host = os.getenv('DB_HOST', 'db')
    db_port = os.getenv('DB_PORT', '3306')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_pass = os.getenv('DB_PASS')

    # 3) 필수 변수 체크
    if not all([db_name, db_user, db_pass]):
        raise ValueError('DB_NAME, DB_USER, DB_PASS 환경변수가 설정되지 않았습니다.')

    # 4) SQLAlchemy 접속 URL 생성 (pymysql 드라이버 사용)
    #    필요한 패키지: sqlalchemy, pymysql
    url = f'mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'

    # 5) 엔진 생성 및 쿼리 실행
    engine = create_engine(url)
    query = 'SELECT * FROM candles'
    df = pd.read_sql(query, con=engine)

    return df