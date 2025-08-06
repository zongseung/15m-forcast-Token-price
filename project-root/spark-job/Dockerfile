# spark-job/Dockerfile
FROM bitnami/spark:latest
WORKDIR /app
RUN mkdir -p /tmp/.ivy2

# ① spark-submit 이 그냥 통하도록 PATH에 bitnami spark bin 추가
ENV PATH="/opt/bitnami/spark/bin:${PATH}"

# 파이썬 의존 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MySQL 커넥터 JAR: 
# Dockerfile 내 COPY 대신, 아래 docker-compose.yml 에서 볼륨 마운트해도 무방합니다.
COPY libs/mysql-connector-j-8.3.0.jar /opt/bitnami/spark/jars/

# 애플리케이션 코드 복사
COPY src/aggregate_jobs.py /app/src/aggregate_jobs.py

# ② 이제 spark-submit 이 PATH에 올라와 있으니 그냥 호출 가능
ENTRYPOINT ["spark-submit", "--master", "local[*]", "/app/src/aggregate_jobs.py"]