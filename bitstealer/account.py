from datetime import datetime
from bitstealer.utils import *
from .models import *


class Account:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        # account_info = request_binance('v3/account',api_key=self.api_key, api_secret=self.api_secret)
        # self.makerCommission= account_info['makerCommission']
        # self.takerCommission= account_info['takerCommission']
        # self.buyerCommission= account_info['buyerCommission']
        # self.sellerCommission= account_info['sellerCommission']
        # self.canTrade= account_info['canTrade']
        # self.canWithdraw= account_info['canWithdraw']
        # self.canDeposit= account_info['canDeposit']
        # self.updateTime= account_info['updateTime']
        # self.balances = {}
        # for i in account_info['balances']:
        #     self.balances[i['asset']] = i['free']

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
        print(response)
        Order(symbol=symbol, order_id=response['orderId'], status=response['status'], price=order_price,
              qty=float(response['executedQty']),
              time=datetime.utcfromtimestamp(response['transactTime'] / 1000).strftime('%Y-%m-%d %H:%M:%S'), side=side,
              low=low, high=high).save()
        return response

    def last_order(self, symbol):
        order = Order.objects.filter(symbol=symbol).last()
        return order

    def cancel_order(self, symbol, orderId):
        params = {
            'symbol': symbol,
            'orderId': orderId
        }
        return request_binance('v3/order', params=params, auth=True, http='DELETE')
