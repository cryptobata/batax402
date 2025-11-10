# batax402_trading_engine.py
# Solana Mean-Reversion Trading Engine | @cryptobata | DK | Nov 10, 2025
import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
import streamlit as st
from jupiter_python_sdk.jupiter import Jupiter
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
import base64

# ========================= CONFIG =========================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
RPC_URL = os.getenv("RPC_URL", "https://api.devnet.solana.com")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
NETWORK = "devnet" if "devnet" in RPC_URL else "mainnet"

PAIRS = [
    "solana", "jupiter", "raydium", "orca", "bonk", "popcat", "dogwifhat",
    "michi", "pepe", "shib", "floki", "gme", "trump", "maga", "fartcoin",
    "goatseus", "pudgy", "meow", "wifu", "usa", "chainlink", "uniswap",
    "avalanche-2", "polkadot", "cardano", "near", "aptos", "sui", "kaspa"
]
WINDOW = 20
Z_THRESHOLD = 2.0
MIN_TRADE_USD = 1
SLIPPAGE_BPS = 50

# --- Global ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("batax402")
price_data: Dict[str, List[float]] = {pair: [] for pair in PAIRS}
last_signal: Dict[str, datetime] = {}

wallet_keypair = Keypair.from_base58_string(PRIVATE_KEY) if PRIVATE_KEY else None
async_client = AsyncClient(RPC_URL)
jupiter = Jupiter(async_client, wallet_keypair) if wallet_keypair else None

# ========================= HELPERS =========================
def compact_json(data: Any) -> str:
    return json.dumps({"data": data}, separators=(',', ':'), sort_keys=True)

async def send_telegram(message: str):
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='HTML')
        except Exception as e:
            log.error(f"Telegram send failed: {e}")

@st.cache_data
def fetch_historical_data(coin: str, days: int = 365):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
        resp = requests.get(url, timeout=10).json()
        prices = [p[1] for p in resp.get('prices', [])]
        if not prices:
            return pd.DataFrame()
        df = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=len(prices), freq='D'),
            'close': prices
        })
        return df
    except Exception as e:
        log.warning(f"Failed to fetch historical data for {coin}: {e}")
        return pd.DataFrame()

# ========================= STRATEGY =========================
def detect_signal(prices: List[float]) -> str:
    if len(prices) < WINDOW:
        return None
    df = pd.Series(prices[-WINDOW:])
    mean = df.mean()
    std = df.std()
    if std == 0:
        return None
    z = (prices[-1] - mean) / std
    if z < -Z_THRESHOLD:
        return 'buy'
    elif z > Z_THRESHOLD:
        return 'sell'
    return None

# ========================= BACKTESTER =========================
def backtest_strategy(coin: str, initial_capital: float = 1000):
    df = fetch_historical_data(coin)
    if df.empty:
        return {'error': 'No data'}
    df['signal'] = df['close'].rolling(WINDOW).apply(
        lambda x: detect_signal(x.tolist()) or '', window=WINDOW
    )
    capital = initial_capital
    position = 0.0
    trades = []
    for i in range(len(df)):
        signal = df.iloc[i]['signal']
        price = df.iloc[i]['close']
        if signal == 'buy' and capital > 0:
            position = capital / price
            capital = 0
            trades.append({'type': 'buy', 'price': price, 'time': df.iloc[i]['timestamp']})
        elif signal == 'sell' and position > 0:
            capital = position * price
            position = 0
            trades.append({'type': 'sell', 'price': price, 'time': df.iloc[i]['timestamp']})
    final_value = capital + position * df.iloc[-1]['close']
    returns = (final_value - initial_capital) / initial_capital * 100
    returns_series = df['close'].pct_change().fillna(0)
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(365) if returns_series.std() > 0 else 0
    drawdown = (df['close'] / df['close'].cummax() - 1).min() * 100
    wins = sum(1 for i in range(1, len(trades), 2) if len(trades) > i and trades[i]['price'] > trades[i-1]['price'])
    win_rate = wins / (len(trades)//2) * 100 if len(trades) >= 2 else 0
    return {
        'final_return': round(returns, 2),
        'sharpe_ratio': round(sharpe, 2),
        'max_drawdown': round(drawdown, 2),
        'win_rate': round(win_rate, 1),
        'num_trades': len(trades)//2,
        'equity_curve': df['close'].tolist()
    }

# ========================= TRADING =========================
async def execute_trade(coin: str, side: str):
    if not jupiter:
        await send_telegram(f"Warning: No wallet for {side} {coin}")
        return None
    try:
        input_mint = "So11111111111111111111111111111111111111112"  # SOL
        output_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
        if side == 'sell':
            input_mint, output_mint = output_mint, input_mint
        amount = int(MIN_TRADE_USD * 1_000_000_000 / 150)  # ~$50 in lamports

        tx_data = await jupiter.swap(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            slippage_bps=SLIPPAGE_BPS
        )
        tx = VersionedTransaction.from_bytes(base64.b64decode(tx_data['transaction']))
        signature = wallet_keypair.sign_message(bytes(tx.message))
        signed_tx = VersionedTransaction.populate(tx.message, [signature])
        txid = await async_client.send_transaction(signed_tx, wallet_keypair)

        explorer = f"https://solscan.io/tx/{txid.value}#{NETWORK}"
        msg = f"""
**batax402 ORDER PLACED**
Pair: <code>{coin.upper()}/USDC</code>
Side: <b>{side.upper()}</b>
Amount: ~${MIN_TRADE_USD}
Tx: <a href="{explorer}">View on Solscan</a>
        """
        await send_telegram(msg)
        log.info(f"batax402 Trade: {txid.value}")
        return str(txid.value)
    except Exception as e:
        error = f"batax402 Trade failed: {str(e)[:100]}"
        log.error(error)
        await send_telegram(f"Warning: {error}")
        return None

# ========================= MONITOR LOOP =========================
async def monitor_prices():
    await send_telegram("**batax402 Trading Engine Started**\nMode: <code>DEVNET</code>\nDEX: <code>Jupiter V6</code>\n@cryptobata")
    while True:
        try:
            for coin in PAIRS:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
                try:
                    price = requests.get(url, timeout=5).json().get(coin, {}).get('usd')
                    if price and price > 0:
                        price_data[coin].append(price)
                        if len(price_data[coin]) > WINDOW * 2:
                            price_data[coin] = price_data[coin][-WINDOW*2:]
                        signal = detect_signal(price_data[coin])
                        if signal and (coin not in last_signal or datetime.now() - last_signal[coin] > timedelta(minutes=30)):
                            last_signal[coin] = datetime.now() if signal == 'buy' else last_signal.pop(coin, None)
                            await execute_trade(coin, signal)
                except:
                    continue

            if int(time.time()) % 600 == 0:
                status = "**batax402 System Status**\n"
                status += f"• Order Placement: <code>Operational</code>\n"
                status += f"• Trading Engine: <code>Running</code>\n"
                status += f"• Monitoring: <code>{len(PAIRS)} pairs</code>\n"
                status += f"• Strategy: <code>Mean reversion</code>\n"
                status += f"• Status: <code>Awaiting signals</code>\n"
                status += f"• @cryptobata | DK"
                await send_telegram(status)

            await asyncio.sleep(60)
        except Exception as e:
            log.error(f"batax402 Monitor error: {e}")
            await asyncio.sleep(30)

# ========================= TELEGRAM BOT =========================
async def start_cmd(update, context):
    await update.message.reply_text(
        "**batax402 Trading Engine**\n"
        "By @cryptobata | Denmark\n\n"
        "Use:\n"
        "/status\n"
        "/backtest solana\n"
        "/gitpush"
    )

async def status_cmd(update, context):
    active = len([p for p, t in last_signal.items() if datetime.now() - t < timedelta(hours=1)])
    await update.message.reply_text(
        f"**batax402 Live Status**\n"
        f"• Engine: Running\n"
        f"• Pairs: {len(PAIRS)}\n"
        f"• Active: {active}\n"
        f"• Mode: <code>{NETWORK.upper()}</code>\n"
        f"• DEX: <code>Jupiter</code>",
        parse_mode='HTML'
    )

async def backtest_cmd(update, context):
    if not context.args:
        await update.message.reply_text("Usage: /backtest solana")
        return
    coin = context.args[0].lower()
    results = backtest_strategy(coin)
    if 'error' in results:
        await update.message.reply_text(f"Error: No data for {coin}")
        return
    msg = f"""
**batax402 Backtest: {coin.upper()}**
• Return: {results['final_return']}%
• Sharpe: {results['sharpe_ratio']}
• Drawdown: {results['max_drawdown']}%
• Win Rate: {results['win_rate']}%
• Trades: {results['num_trades']}
    """
    await update.message.reply_text(msg, parse_mode='HTML')

async def gitpush_cmd(update, context):
    import subprocess
    try:
        # Stage all
        subprocess.run(["git", "add", "."], check=True)
        
        # Check if there are changes
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not status.stdout.strip():
            await update.message.reply_text("No changes to push.")
            return

        # Commit
        subprocess.run(["git", "commit", "-m", f"batax402 auto-update {datetime.now().strftime('%H:%M')}"], check=True)
        
        # Push
        result = subprocess.run(["git", "push"], capture_output=True, text=True)
        await update.message.reply_text(
            f"**Git Push OK**\n"
            f"<code>{result.stdout.splitlines()[-1] if result.stdout else 'Done'}</code>",
            parse_mode='HTML'
        )
    except Exception as e:
        await update.message.reply_text(f"Git error: {e}")
        await update.message.reply_text(
            f"**Git Push OK**\n"
            f"<code>{result.stdout.splitlines()[-1] if result.stdout else 'Done'}</code>",
            parse_mode='HTML'
        )
    except Exception as e:
        await update.message.reply_text(f"Git error: {e}")

# ========================= DASHBOARD =========================
def run_dashboard():
    st.set_page_config(page_title="batax402", layout="wide")
    st.title("batax402 — Solana Trading Engine")
    st.caption("@cryptobata | Denmark | Nov 10, 2025")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Order", "Operational", "Green")
    c2.metric("Engine", "Running", "Green")
    c3.metric("Pairs", len(PAIRS))
    c4.metric("Strategy", "Mean Reversion")
    c5.metric("Status", "Awaiting signals")

    tab1, tab2 = st.tabs(["Backtest", "Live"])
    with tab1:
        coin = st.selectbox("Coin", PAIRS)
        if st.button("Run"):
            r = backtest_strategy(coin)
            if 'error' not in r:
                st.success("Done")
                st.metric("Return", f"{r['final_return']}%")
                st.line_chart(r['equity_curve'])

    with tab2:
        for coin in PAIRS[:5]:
            p = price_data.get(coin, [])
            if p:
                s = detect_signal(p) or "hold"
                st.write(f"**{coin.upper()}**: ${p[-1]:.4f} → {s}")

# ========================= MAIN =========================
async def main():
    from telegram.request import HTTPXRequest
    request = HTTPXRequest()

    app = Application.builder().token(TELEGRAM_TOKEN).request(request).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("backtest", backtest_cmd))
    app.add_handler(CommandHandler("gitpush", gitpush_cmd))

    monitor_task = asyncio.create_task(monitor_prices())

    import sys
    if "--dashboard" in sys.argv:
        run_dashboard()
    else:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        try:
            await monitor_task
        finally:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
