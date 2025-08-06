import os
import json
import joblib
import torch
import discord
import matplotlib.pyplot as plt
from discord.ext import commands, tasks
from config import CHANNEL_ID, MODEL_PATH, SCALER_PATH
from utils import load_data, forecast_next_hour
from src.model import Seq2Seq

class ForecastCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.channel = None

        # ─── 하이퍼파라미터 로드 ────────────────────────────────────────────
        params_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir, "model", "config", "best_params.json"
        )
        with open(params_path, "r") as f:
            bp = json.load(f)

        self.feature_cols = [
            'Open','High','Low','Volume','total_transaction',
            'SMA','Upper_Band','Lower_Band',
            'MACD','MACD_Signal','RSI','EMA_50','EMA_200'
        ]

        # ─── 모델 초기화 & 로드 ────────────────────────────────────────────
        self.model = Seq2Seq(
            enc_input_dim=len(self.feature_cols),
            dec_input_dim=1,
            hid_dim=bp['hid_dim'],
            out_len=4,
            n_layers=bp['n_layers'],
            dropout=bp.get('lstm_dropout_rate', 0.0)
        )
        state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        self.model.load_state_dict(state)
        self.model.eval()

        # ─── 스케일러 로드 ─────────────────────────────────────────────────
        scalers = joblib.load(SCALER_PATH)
        self.scaler_X, self.scaler_y = scalers["scaler_X"], scalers["scaler_y"]

        # ─── 정시마다 실행되는 태스크 시작 ───────────────────────────────────
        self.hourly_task.start()

    @tasks.loop(hours=1)
    async def hourly_task(self):
        # 채널 객체 한 번만 가져오기
        if self.channel is None:
            self.channel = self.bot.get_channel(CHANNEL_ID)
            if self.channel is None:
                return

        # 1) 테스트 세그먼트 그래프 전송
        img_path = "/app/model/output/test_segment_plot.png"
        if not os.path.exists(img_path):
            await self.channel.send(f"❌ 테스트 세그먼트 그래프를 찾을 수 없습니다:\n`{img_path}`")
            return
        await self.channel.send(file=discord.File(img_path))

        # 2) 이전 실제 4포인트 가져오기
        df = load_data()
        prev_actual = df["Close"].values[-4:].tolist()

        # 3) 4스텝 예측값 계산
        future = forecast_next_hour(
            self.model, self.scaler_X, self.scaler_y,
            df, self.feature_cols,
            in_len=1440,
            device=torch.device("cpu")
        )

        # 4) 실제값 & 예측값 전송
        await self.channel.send(f"⏮ Previous actual 4-step: {prev_actual}")
        await self.channel.send(f"🔮 Future 4-step forecast: {future}")

    @commands.command(name="forecast_now")
    async def forecast_now(self, ctx):
        """!forecast_now 입력 시 즉시 테스트 세그먼트 플롯 + 실제/예측값을 보냅니다."""
        await ctx.send("⏳ 예측을 시작합니다…")
        await self.hourly_task()
        await ctx.send("✅ 완료!")

    @commands.command(name="send_test_plot")
    async def send_test_plot(self, ctx):
        """!send_test_plot 입력 시 테스트 세그먼트 플롯만 보냅니다."""
        img = "/app/model/output/test_segment_plot.png"
        if not os.path.exists(img):
            return await ctx.send(f"❌ 파일이 없습니다: `{img}`")
        await ctx.send(file=discord.File(img))

    @hourly_task.before_loop
    async def before_hourly(self):
        await self.bot.wait_until_ready()

async def setup(bot):
    cog = ForecastCog(bot)
    await bot.add_cog(cog)
    # 봇 시작 직후 한 번 실행
    await cog.hourly_task()
