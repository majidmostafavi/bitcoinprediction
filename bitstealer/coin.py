import talib
import numpy as np
from decimal import Decimal
from bitstealer.utils import *
from bitstealer.models import *
from datetime import datetime
from dateutil.relativedelta import relativedelta
import websocket
import json


class Coin:
    def __init__(self, base, quote, qty=1):
        self.coin = base
        self.price = quote
        self.symbol = base + quote
        self.qty = qty
        self.trade_list = []

    def get_trade(self, count=500):
        self.trade_list = request_binance('v1/trades', params={'symbol': self.symbol, 'limit': count})
        return self

    def save_trade(self):

        for i in self.trade_list:
            try:
                item = Trade(symbol=self.symbol, price=i['price'], qty=i['qty'],
                             time=datetime.utcfromtimestamp(i['time'] / 1000), is_buyer_maker=i['isBuyerMaker'])
                item.save()
            except:
                continue
        return self

    def candle_stick(self, interval, limit=500):
        params = {
            'symbol': self.symbol,
            'interval': interval,
            'limit': limit
        }
        return request_binance('v1/klines', params=params)

    def create_candle(self, interval='1h'):
        headers = {'X-CMC_PRO_API_KEY': 'c1dadb4e-bee8-4120-8854-fc4aca0ac645'}
        url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
        result = requests.get(url, headers=headers).json()

        volume_candle = float(0)
        open_candle = 0
        high_candle = 0
        close_candle = 0
        low_candle = 0
        if interval=='1h':
            date_candle = datetime.now() - relativedelta(hours=1)
        if interval=='1d':
            date_candle = datetime.now() - relativedelta(hours=24)
        for i in Trade.objects.filter(time__gt=date_candle):
            volume_candle += float(i.qty) * float(i.price)
            close_candle = i.price
            if open_candle == 0:
                open_candle = i.price
            if i.price > high_candle:
                high_candle = i.price
            if i.price < low_candle or low_candle == 0:
                low_candle = i.price
        if interval=='1h':
            total_market_cap = result['data']['quote']['USD']['total_market_cap']
            altcoin_market_cap = result['data']['quote']['USD']['altcoin_market_cap']
            coin_market_cap = total_market_cap - altcoin_market_cap
            item = HourlyCandle(symbol=self.symbol, date=date_candle, open=open_candle, close=close_candle, low=low_candle,
                            high=high_candle, volume=volume_candle,coin_marketcap=coin_market_cap,marketcap=total_market_cap)
        if interval=='1d':
            total_market_cap = result['data']['quote']['USD']['total_market_cap']
            altcoin_market_cap = result['data']['quote']['USD']['altcoin_market_cap']
            volume_candle = result['data']['quote']['USD']['total_volume_24h']-result['data']['quote']['USD']['altcoin_volume_24h']
            coin_market_cap = total_market_cap - altcoin_market_cap
            item = DailyCandle(symbol=self.symbol, date=date_candle, open=open_candle, close=close_candle, low=low_candle,
                                high=high_candle, volume=volume_candle,coin_marketcap=coin_market_cap,marketcap=total_market_cap)
        item.save()
        # print(selxf.symbol + " " + date_candle.strftime(
        #     '%Y-%m-%d %H:%M') + " " + str(open_candle) + " " + str(close_candle) + " " + str(low_candle) + " " + str(
        #     high_candle) + " " + str(volume_candle))
        return self

    def last_price(self, count=500):
        # trades = self.get_trade(count=count)
        # prices = []
        # for i in trades:
        #     prices.append(float(i['price']))

        return float(request_binance('v3/ticker/price', params={'symbol': self.symbol})['price'])

    def indicators(self, interval):
        candles = self.candle_stick(interval)
        closes = []
        opens = []
        lows = []
        highs = []
        for i in candles:
            closes.append(Decimal(i[4]))
            opens.append(Decimal(i[1]))
            lows.append(Decimal(i[3]))
            highs.append(Decimal(i[2]))

        k, d = talib.STOCH(np.array(highs, dtype='f8'), np.array(lows, dtype='f8'), np.array(closes, dtype='f8'),
                           slowd_period=2)
        sma = talib.SMA(np.array(closes, dtype='f8'), timeperiod=50)
        ema = talib.EMA(np.array(highs, dtype='f8'), timeperiod=15)

        return [k.tolist(), d.tolist(), sma.tolist(), ema.tolist(), closes, highs, lows, opens]

    def on_message(self, message):
        trade=json.loads(message)
        item = Trade(symbol=self.symbol, price=trade['p'], qty=trade['q'],
                     time=datetime.utcfromtimestamp(trade['T'] / 1000), is_buyer_maker=trade['m'])
        item.save()

    def on_error(self, error):
        print(error)

    def on_close(self, ):
        print("### closed ###")

    def streamer(self, method):
        ws = websocket.WebSocketApp("wss://stream.binance.com:9443/ws/" + self.symbol.lower() + "@" + method,
                                    on_message=self.on_message, on_close=self.on_close, on_error=self.on_error)
        ws.run_forever()
