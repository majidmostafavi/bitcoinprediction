from django.core.management.base import BaseCommand
from bitstealer.coin import Coin
from bitstealer.models import *
import requests
import urllib
from datetime import datetime


class Command(BaseCommand):
    help = 'Get 1 Day Recent Candle List'

    def handle(self, *args, **kwargs):
        Coin('BTC','USDT','1').create_candle('1d')
