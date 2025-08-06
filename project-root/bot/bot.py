import discord
from discord.ext import commands
from config import TOKEN

intents = discord.Intents.default()
intents.message_content = True  # ❗️메시지 접근 허용 (Discord API 2022 이후 필수)

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user} (ID: {bot.user.id})")
    await bot.load_extension("cogs.forecast")

@bot.command()
async def ping(ctx):
    await ctx.send("✅ 봇 응답 확인됨")

if __name__ == "__main__":
    bot.run(TOKEN)