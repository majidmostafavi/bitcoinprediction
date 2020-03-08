from django.core.management.base import BaseCommand
from bitstealer.coin import Coin


class Command(BaseCommand):
    help = 'Get Recent Trade List'

    def handle(self, *args, **kwargs):
        Coin('BTC','USDT','1').streamer('trade')
        # websocket.enableTrace(True)


