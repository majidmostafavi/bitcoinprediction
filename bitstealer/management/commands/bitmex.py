from django.core.management.base import BaseCommand
from bitstealer.models import *
from datetime import datetime
import websocket
import json

def on_message(ws, message):
    trade = json.loads(message)
    if(trade["table"]=='trade'):
        for i in trade['data']:
            is_buyer_maker = True if i['side'] == 'Sell' else False
            item = BitmexTrade(symbol="BTCUSDT", price=i['price'], qty=i['size']/i['price'],time=datetime.strptime(i['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ"), is_buyer_maker=is_buyer_maker)
            item.save()


def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")


class Command(BaseCommand):
    help = 'Get Recent Trade List'

    def handle(self, *args, **kwargs):
        ws = websocket.WebSocketApp("wss://www.bitmex.com/realtime?subscribe=trade:XBTUSD",
                                    on_message=on_message, on_close=on_close, on_error=on_error)
        ws.run_forever()