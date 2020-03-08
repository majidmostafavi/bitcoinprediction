from django.core.management.base import BaseCommand
from bitstealer.models import *
from bitstealer.coin import Coin
from bitstealer.utils import *


class Command(BaseCommand):
    help = 'Trade'

    def handle(self, *args, **kwargs):
        mahdi = User.objects.get(id=2)
        account = mahdi.account

        neo = Coin(base='NEO', quote='BTC', qty=3)
        ont = Coin(base='ONT', quote='BTC', qty=25)

        traddder(account,neo)
        traddder(account,ont)
