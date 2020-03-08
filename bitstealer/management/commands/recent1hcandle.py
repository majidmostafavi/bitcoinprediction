from django.core.management.base import BaseCommand
from bitstealer.coin import Coin


class Command(BaseCommand):
    help = 'Get 1 Hour Recent Candle List'

    def handle(self, *args, **kwargs):
        Coin('BTC','USDT','1').create_candle('1h')
        # print(datetime(2019, 10, 20, 13))
        # candles = request_binance('v3/klines', params={'symbol': "BTCUSDT", 'interval': '1h', 'limit': 1000})
        # for candle in candles:
        #     if 1571583600000 < candle[0]<1571644740000:
        #         d = datetime.utcfromtimestamp(candle[0] / 1000)
        #         o = candle[1]
        #         h = candle[2]
        #         l = candle[3]
        #         c = candle[4]
        #         vol = float(candle[5])
        #         avg = (float(o)+float(l)+float(h)+float(c))/4
        #         item = HourlyCandle(symbol=symbol,date=d,open=o,close=c,low=l,high=h,volume=(vol*avg))
        #         item.save()
        #         print(symbol + " " + d.strftime(
        #             '%Y-%m-%d %H:%M') + " " + o + " " + c + " " + l + " " + h + " " + str(vol*avg))


