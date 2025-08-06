#!/usr/bin/env python3
# coding: utf-8

import os
from pyspark.sql import SparkSession
from dotenv import load_dotenv

def main():
    # 1) .env 로드
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path)
    db_url  = os.getenv("DB_URL")    # e.g. jdbc:mysql://db:3306/hype_db
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")

    # 2) SparkSession 생성 (MySQL 커넥터 JAR은 이미 컨테이너 이미지에 포함되어 있다고 가정)
    spark = (
        SparkSession.builder
        .appName("FetchCandles")
        .config("spark.driver.extraClassPath", "/opt/jars/mysql-connector-java-8.3.0.jar")
        .config("spark.jars.ivy", "/tmp/.ivy2")             
        .getOrCreate()
    )


    # 3) MySQL 'candles' 테이블을 DataFrame으로 읽어오기
    candles_df = (spark.read
                  .format("jdbc")
                  .option("url",      db_url)
                  .option("dbtable",  "candles")
                  .option("user",     db_user)
                  .option("password", db_pass)
                  .option("driver",   "com.mysql.cj.jdbc.Driver")
                  .load())
    spark.stop()

if __name__ == "__main__":
    main()