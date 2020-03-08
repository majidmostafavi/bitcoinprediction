from django.core.management.base import BaseCommand
from django.db.models import Q
import datetime
from datetime import timedelta
from bitstealer.models import *


class Command(BaseCommand):
    help = 'Test'

    def handle(self, *args, **kwargs):
        # start_date = datetime.datetime(2020, 1, 6, 9, 22) #TODO BINACNE STARTED DATE
        start_date = datetime.datetime(2020, 1, 6, 10, 32)
        while start_date < datetime.datetime.now():
            end_date = start_date + timedelta(minutes=1)
            candles = BitmexTrade.objects.filter(time__range=(start_date, end_date))
            volume_candle = float(0)
            open_candle = 0
            high_candle = 0
            close_candle = 0
            low_candle = 0
            for i in candles:
                volume_candle += float(i.qty) * float(i.price)
                close_candle = i.price
                if open_candle == 0:
                    open_candle = i.price
                if i.price > high_candle:
                    high_candle = i.price
                if i.price < low_candle or low_candle == 0:
                    low_candle = i.price
            item = BitmxMinCandle(symbol="BTCUSDT", date=start_date, open=open_candle, close=close_candle, low=low_candle,
                                high=high_candle, volume=volume_candle,coin_marketcap=None,marketcap=None)
            item.save()
            start_date = end_date
