import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 불러오기

# Discord bot token (정상적으로 환경변수에서 읽어옴)
TOKEN = os.getenv("DISCORD_TOKEN")

# Discord 채널 ID (문자열이 아닌 int로 변환)
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "1329442048750784552"))

# 모델 경로들
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/output/model.pth")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/model/output/scaler.pkl")