# batax402_sniper.py — MEV Sniping Bot | @cryptobata | DK
import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from telegram import Bot
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from jupiter_python_sdk.jupiter import Jupiter
import requests
import json

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
RPC_URL = os.getenv("RPC_URL", "https://api.devnet.solana.com")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("batax402_sniper")

async_client = AsyncClient(RPC_URL)
wallet_keypair = Keypair.from_base58_string(PRIVATE_KEY)
jupiter = Jupiter(async_client, wallet_keypair)

RAYDIUM_API = "https://api.raydium.io/v2/sdk/liquidity/mainnet.json"  # Change to devnet for testing

async def send_telegram(message):
    if TELEGRAM_TOKEN and CHAT_ID:
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='HTML')

async def is_honeypot(token):
    try:
        url = f"https://api.honeypot.is/v2/IsHoneypot?address={token}"
        r = requests.get(url, timeout=5).json()
        return r['IsHoneypot']
    except:
        return True

async def snipe_new_pool():
    await send_telegram("**batax402 SNIPER STARTED**\nScanning Raydium for new pools...\n@cryptobata")
    seen_pools = set()
    while True:
        try:
            r = requests.get(RAYDIUM_API, timeout=10).json()
            pools = r['official'] + r['unOfficial']
            for pool in pools:
                pool_id = pool['id']
                if pool_id in seen_pools:
                    continue
                seen_pools.add(pool_id)
                base = pool['baseMint']
                quote = pool['quoteMint']
                liquidity = pool['baseVault']  # Rough check
                if quote != "So11111111111111111111111111111111111111112":  # Not SOL pair
                    continue
                if int(liquidity) < 1000000000:  # <1 SOL liquidity
                    continue
                if is_honeypot(base):
                    continue

                # SNIPE $1
                amount = int(1 * 1_000_000_000 / 150)  # $1 in lamports
                tx_data = await jupiter.swap(
                    input_mint="So11111111111111111111111111111111111111112",
                    output_mint=base,
                    amount=amount,
                    slippage_bps=100
                )
                tx = VersionedTransaction.from_bytes(base64.b64decode(tx_data['transaction']))
                signature = wallet_keypair.sign_message(bytes(tx.message))
                signed_tx = VersionedTransaction.populate(tx.message, [signature])
                txid = await async_client.send_transaction(signed_tx, wallet_keypair)

                explorer = f"https://solscan.io/tx/{txid.value}"
                msg = f"""
**SNIPED NEW TOKEN!**
Token: <code>{base}</code>
Pool: <code>{pool_id}</code>
Amount: ~$1
Tx: <a href="{explorer}">View</a>
@cryptobata
                """
                await send_telegram(msg)
                log.info(f"SNIPED {base}")
        except Exception as e:
            log.error(f"Snipe error: {e}")
        await asyncio.sleep(5)  # Scan every 5 sec

if __name__ == "__main__":
    asyncio.run(snipe_new_pool())