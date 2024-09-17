import tushare as ts
import pandas as pd
from datetime import datetime, timedelta,timezone
import yaml
import requests as req
from binance.client import Client


def get_client():
    with open("config/binance_api.yaml", "r", encoding="utf-8") as file:
          config = yaml.safe_load(file)
    key = config['binance_api']['api_key']
    secret = config['binance_api']['api_secret']
    return Client(key, secret)

def get_klines_dataframe(lookback_duration,client=get_client(), symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, ): 
    """
    获取币安历史K线数据，从当前时间往回推指定的时间段，并将其转换为 pandas DataFrame。

    参数:
        client: Binance API 客户端实例
        symbol: 交易对，例如 'BTCUSDT'
        interval: K线时间间隔，例如 '1h', '1d', '5m' 等
        lookback_duration: 要往回推的时间段，可以是 timedelta 对象，例如 timedelta(days=7) 表示过去7天

    返回值:
        pandas DataFrame，包含开盘价、最高价、最低价、收盘价、交易量等信息
    """
    # 计算当前时间和回溯的开始时间
    end_time = datetime.now(timezone.utc)
    start_time = end_time - lookback_duration  # 向过去回溯的时间
    # 将时间格式化为字符串，供API使用
    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    # 获取K线数据
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    # 将K线数据转换为DataFrame，并为每列添加合适的列名
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    # 将时间戳转换为可读日期时间格式
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    # 将价格和交易量列转换为浮点数
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['quote_asset_volume'] = df['quote_asset_volume'].astype(float)
    df['taker_buy_base_asset_volume'] = df['taker_buy_base_asset_volume'].astype(float)
    df['taker_buy_quote_asset_volume'] = df['taker_buy_quote_asset_volume'].astype(float)
    # 删除无用的列 'Ignore'
    df = df.drop(columns=['ignore'])
    return df


def data_loader(symbol='BTCUSDT',interval:str='1h',back_days:int=7):
     """
     获取币安历史数据，返回DataFrame
     symbol: 交易对，例如 'BTCUSDT'
     interval: K线时间间隔，例如 '1h', '1d', '5m' 等
     back_days: 要往回推的时间段，单位：天（默认7天）
     """
     client = get_client()
     match interval:
        case '1m':  # 1分钟
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, lookback_duration=timedelta(days=back_days))
        case '3m':  # 3分钟
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_3MINUTE, lookback_duration=timedelta(days=back_days))
        case '5m':  # 5分钟
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE, lookback_duration=timedelta(days=back_days))
        case '15m':  # 15分钟
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, lookback_duration=timedelta(days=back_days))
        case '30m':  # 30分钟
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_30MINUTE, lookback_duration=timedelta(days=back_days))
        case '1h':  # 1小时
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, lookback_duration=timedelta(days=back_days))
        case '2h':  # 2小时
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_2HOUR, lookback_duration=timedelta(days=back_days))
        case '4h':  # 4小时
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_4HOUR, lookback_duration=timedelta(days=back_days))
        case '6h':  # 6小时
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_6HOUR, lookback_duration=timedelta(days=back_days))
        case '8h':  # 8小时
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_8HOUR, lookback_duration=timedelta(days=back_days))
        case '12h':  # 12小时
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_12HOUR, lookback_duration=timedelta(days=back_days))
        case '1d':  # 1天
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, lookback_duration=timedelta(days=back_days))
        case '3d':  # 3天
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_3DAY, lookback_duration=timedelta(days=back_days))
        case '1w':  # 1周
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_1WEEK, lookback_duration=timedelta(weeks=back_days // 7))
        case '1M':  # 1月
            df = get_klines_dataframe(client=client, symbol=symbol, interval=Client.KLINE_INTERVAL_1MONTH, lookback_duration=timedelta(weeks=back_days // 30))
        case _:  # 默认情况，处理未知的时间间隔
            raise ValueError(f"Unsupported interval: {interval}")
     return df


from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET, TIME_IN_FORCE_GTC
from binance.exceptions import BinanceAPIException

def market_buy_btc_with_usdt(client=get_client(), usdt_amount:int=100, symbol='BTCUSDT'):
    """
    市价买入指定金额的 BTC.
    
    :param client: Binance API 客户端实例
    :param usdt_amount: 购买 BTC 的 USDT 数量
    :param symbol: 交易对 (默认为 'BTCUSDT')
    
    :return: 成功返回订单信息，失败返回 None
    """
    try:
        # 获取当前市场价格
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        # 获取交易对的 LOT_SIZE 限制
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
        min_qty = float(lot_size_filter['minQty'])  # 最小交易数量
        step_size = float(lot_size_filter['stepSize'])  # 步长
        # 计算买入的 BTC 数量
        quantity = usdt_amount / current_price
        # 使用 step_size 调整数量，确保符合 Binance 的限制（向下取整至步长的倍数）
        quantity = quantity - (quantity % step_size)
        quantity = round(quantity, 8)  # BTC 数量限制到 8 位小数
        # 确保数量大于最小交易数量
        if quantity < min_qty:
            print(f"计算后的买入数量 {quantity} 小于最小交易数量 {min_qty}")
            return None
        # 创建市价买入订单
        order = client.create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=quantity  # 买入的 BTC 数量
        )
        return order  # 返回订单信息
    except BinanceAPIException as e:
        print(f"市价买入现货失败: {e}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None