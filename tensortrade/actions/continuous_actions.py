# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np

from typing import Union
from gym.spaces import Box

from tensortrade.actions import ActionScheme, TradeActionUnion, DTypeString
from tensortrade.trades import Trade, TradeType


class ContinuousActions(ActionScheme):
    """Simple continuous action scheme, which calculates the trade amount as
    a fraction of the total balance.

    Arguments:
        max_allowed_slippage_percent: The maximum amount above the current price the scheme will pay for an instrument.
            Defaults to 1.0 (i.e. 1%).
        instrument: A `str` designating the instrument to be traded.
            Defaults to 'BTC'.
        dtype: A `type` or `str` corresponding to the dtype of the `action_space`.
            Defaults to `np.float32`.
    """

    def __init__(self,
                 instrument: str = 'BTC',
                 max_allowed_slippage_percent: float = 0,
                 dtype: DTypeString = np.float32,
                 max_allowed_amount: int = 1000,
                 max_allowed_amount_percent: int = 1
                 ):
        super().__init__(action_space=Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=dtype), dtype=dtype)

        self._instrument = self.context.get('instruments', instrument)
        self.max_allowed_slippage_percent = self.context.get('max_allowed_slippage_percent', None) or \
            max_allowed_slippage_percent
        self.max_allowed_amount = max_allowed_amount
        self.max_allowed_amount_percent = max_allowed_amount_percent

        if isinstance(self._instrument, list):
            self._instrument = self._instrument[0]

    def get_trade(self, current_step: int, action: TradeActionUnion) -> Trade:
        trade_amount = abs(float(action))

        if np.sign(action) > 0:
            trade_type = TradeType(1)
        elif np.sign(action) < 0:
            trade_type = TradeType(2)
        else:
            trade_type = TradeType(0)

        current_price = self._exchange.current_price(symbol=self._instrument)
        # base_precision = self._exchange.base_precision
        instrument_precision = self._exchange.instrument_precision
        amount = round(trade_amount * self.max_allowed_amount, instrument_precision)
        price = current_price

        # amount = self._exchange.instrument_balance(self._instrument)
        # if trade_type is TradeType.MARKET_LONG:
        #     price_adjustment = 1 + (self.max_allowed_slippage_percent / 100)
        #     price = max(round(current_price * price_adjustment, base_precision), base_precision)
        #     # amount = round(self._exchange.balance * 0.99 *
        #     #                trade_amount / price, instrument_precision)
        #
        #     amount = round(trade_amount * self.max_allowed_amount, instrument_precision)
        #     # amount = round(trade_amount * (self._exchange.balance * self._exchange.leverage * price) * self.max_allowed_amount_percent / 100, instrument_precision)
        #
        # elif trade_type is TradeType.MARKET_SHORT:
        #     price_adjustment = 1 - (self.max_allowed_slippage_percent / 100)
        #     price = round(current_price * price_adjustment, base_precision)
        #     # amount_held = self._exchange.portfolio.get(self._instrument, 0)
        #     # amount = round(amount_held * trade_amount, instrument_precision)
        #
        #     amount = round(trade_amount * self.max_allowed_amount, instrument_precision)
        #     # amount = round(trade_amount * (self._exchange.balance * self._exchange.leverage * price) * self.max_allowed_amount_percent / 100, instrument_precision)

        return Trade(current_step, self._instrument, trade_type, amount, price)
