from django.core.management.base import BaseCommand
from django.core.cache import cache
from bitstealer.models import *
from bitstealer.utils import *
from datetime import datetime



class Command(BaseCommand):
    help = 'Renew Exchange Info'

    def handle(self, *args, **kwargs):
        exchangeInfo = request_binance(method='v1/exchangeInfo')
        for i in exchangeInfo['symbols']:
            for j in i['filters']:
                if(j['filterType']=='PRICE_FILTER'):
                    ExchangeInfo(symbol=i['symbol'],status=i['status'],baseAsset=i['baseAsset'],quoteAsset=i['quoteAsset'],tickSize=j['tickSize']).save()