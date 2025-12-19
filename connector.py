import ccxt
import pandas as pd

CRYPTOS = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "XRP": "XRP/USDT", "SOL": "SOL/USDT"}

def get_data(coin_key, interval="1h"):
    symbol = CRYPTOS.get(coin_key, "BTC/USDT")
    exchange = ccxt.kucoin()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=200)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df.rename(columns={'close': 'price'})
    except:
        return None
