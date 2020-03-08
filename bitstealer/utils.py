import hashlib
import hmac
import urllib
import requests
import time
from stem import Signal
from stem.control import Controller
from datetime import datetime
from bitstealer.models import Order



# def refresh_tor():
#     with Controller.from_port(port=9050) as controller:
#         controller.authenticate()
#         print('Tor is running version %s' % controller.get_version())
#         controller.signal(Signal.NEWNYM)
#
#
# def get_tor_session():
#     session = requests.session()
#     # Tor uses the 9050 port as the default socks port
#     session.proxies = {'http': 'socks5://127.0.0.1:9050',
#                        'https': 'socks5://127.0.0.1:9050'}
#     return session


def get_query_string_with_signature(params, api_secret):
    api_secret = api_secret
    querystring = urllib.parse.urlencode(params)
    params['signature'] = hmac.new(api_secret.encode('utf-8'), querystring.encode('utf-8'), hashlib.sha256).hexdigest()
    return urllib.parse.urlencode(params)


def request_binance(method, api_key=None, api_secret=None, params={}, http='GET'):
    base_url = 'https://api.binance.com/api/'
    url = base_url + method + "?"
    headers = {}
    if api_key and api_secret:
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = 5000
        url += get_query_string_with_signature(params, api_secret=api_secret)
        headers = {'X-MBX-APIKEY': api_key}
    else:
        url += urllib.parse.urlencode(params)
    if http == 'GET':
        return requests.get(url, headers=headers).json()
    if http == 'POST':
        return requests.post(base_url + method, data=params, headers=headers).json()
    if http == 'DELETE':
        return requests.delete(base_url + method, data=params, headers=headers).json()



def open_order(self, symbol=None):
    params = {}
    if symbol:
        params['symbol'] = symbol
    return request_binance('v3/openOrders', params=params, api_key=self.api_key, api_secret=self.api_secret)


def balance(self, coin):
    return self.balances[coin]


def send_order(self, symbol, side, price, typ, quantity, high, low, time_in_force=None, stop_price=0.0):
    params = {
        'symbol': symbol,
        'side': side,
        'price': price,
        'type': typ,
        'quantity': quantity,
        # 'icebergQty':'LIMIT',
        'newOrderRespType': 'FULL'
    }
    if stop_price > 0:
        params['stopPrice'] = stop_price
    if time_in_force:
        params['timeInForce'] = time_in_force

    response = request_binance('v3/order', params=params, api_key=self.api_key, api_secret=self.api_secret,
                               http='POST')
    order_price = 0
    for i in response['fills']:
        order_price += float(i['price']) * float(i['qty'])
    order_price = order_price / quantity
    Order(symbol=symbol, order_id=response['orderId'], account=self, status=response['status'], price=order_price,
          qty=float(response['executedQty']),
          time=datetime.utcfromtimestamp(response['transactTime'] / 1000).strftime('%Y-%m-%d %H:%M:%S'), side=side,
          low=low, high=high).save()
    return response


def last_order(self, symbol):
    order = Order.objects.filter(account=self, symbol=symbol).last()
    if order.status == 'FILLED':
        return order
    else:
        params = {
            'symbol': order.symbol,
            'orderId': order.order_id
        }
        response = request_binance('v3/order', params=params, api_key=self.api_key, api_secret=self.api_secret)
        order.status = response['status']
        order.price = response['price']
        order.qty=float(response['executedQty'])
        order.time = datetime.utcfromtimestamp(response['time'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        order.save()
        return order

def cancel_order(self, symbol, orderId):
    params = {
        'symbol': symbol,
        'orderId': orderId
    }
    return request_binance('v3/order', params=params, auth=True, http='DELETE')


def order(A):
    if A == sorted(A, reverse=False):
        return 1
    elif A == sorted(A, reverse=True):
        return -1
    else:
        return 0


def average(list):
    return sum(list) / len(list)


def send_order(symbol, side, price, typ, quantity, timeInForce=None, stopPrice=0.0):
    params = {
        'symbol': symbol,
        'side': side,
        'price': price,
        'type': typ,
        'quantity': quantity,
        # 'icebergQty':'LIMIT',
        'newOrderRespType': 'FULL'
    }
    if stopPrice > 0:
        params['stopPrice'] = stopPrice
    if timeInForce:
        params['timeInForce'] = timeInForce
    return request_binance('v3/order', params, auth=True, http='POST')


def find_last_summit(lis):
    ind = 0
    for i, e in reversed(list(enumerate(lis))):
        if e > 55 and ind == 0:
            ind = i
    return ind - 1


def find_last_valley(lis):
    ind = 0
    for i, e in reversed(list(enumerate(lis))):
        if e < 45 and ind == 0:
            ind = i
    return ind - 1


def tick_size(coin):
    params = {
        "NEO": 0.000001,
        "ONT": 0.0000001
    }
    return params[coin]


def traddder(account, coin):

    print(f'{coin.symbol}:')
    k1m, d1m, sma1m, ema1m, closes, highs, lows, opens = coin.indicators('1m')
    #k3m, d3m, sma3m, ema3m, _, _, _, _ = coin.indicators('3m')
    k15m, d15m, sma15m, ema15m, _, _, _, _ = coin.indicators('15m')
    k1h, d1h, sma1h, ema1h, _, _, _, _ = coin.indicators('1h')
    ticksize = tick_size(coin.coin)
    lp = coin.last_price()
    lo = account.last_order(coin.symbol)
    print(f'Side :{lo.side}')
    oo = account.open_order(coin.symbol)
    isummit = find_last_summit(d1h)
    ivalley = find_last_valley(d1h)
    print(f'last price: {lp} ')
    if len(oo)==0 :
        if lo.side == 'BUY':
            if lp < 0.99*float(lo.low):
                print(f"Sell STOP LOSS")
                account.send_order(symbol=coin.symbol, side='SELL', typ='LIMIT', time_in_force='GTC',
                                   price=lp - (1 * ticksize), quantity=coin.qty, high=highs[-2], low=lows[-2])
            else:
                if d1h[-1] < 55:
                    if isummit > ivalley:
                        print(f'k1m : {isummit} ')
                        if (((float(lo.price)/lp)-1)*100)>2:
                            print(f'{coin.symbol} Sell')
                            account.send_order(symbol=coin.symbol, side='SELL', typ='LIMIT', time_in_force='GTC',
                                               price=lp - ticksize, quantity=coin.qty, high=highs[-2], low=lows[-2])
                        else:
                            if lp<highs[-2]:
                                account.send_order(symbol=coin.symbol, side='SELL', typ='LIMIT', time_in_force='GTC',
                                                   price=lp - ticksize, quantity=coin.qty, high=highs[-2], low=lows[-2])
        else:
            if sma15m[-1]>sma15m[-2]:
                if d15m > 45:
                    print(f'{coin.symbol} Buy')
                    account.send_order(symbol=coin.symbol, side='BUY', typ='LIMIT', time_in_force='GTC',
                                       price=lp + (1 * ticksize), quantity=coin.qty, high=highs[-2],
                                       low=lows[-2])
    print(' ')



