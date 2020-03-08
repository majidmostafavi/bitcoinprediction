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
import pandas as pd

import tensortrade.slippage as slippage

from gym.spaces import Space, Box
from typing import List, Dict

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges import Exchange
from tensortrade.features import FeaturePipeline


class BitmexExchange3(Exchange):
    """An exchange, in which the price history is based off the supplied data frame and
    trade execution is largely decided by the designated slippage model.

    If the `data_frame` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environments.
    """

    def __init__(self, data_frame: pd.DataFrame = None, **kwargs):
        super().__init__(
            dtype=self.default('dtype', np.float32),
            feature_pipeline=self.default('feature_pipeline', None),
            **kwargs
        )

        self._commission_percent = self.default('commission_percent', 0.075, kwargs)
        self._base_precision = self.default('base_precision', 5, kwargs)
        self._instrument_precision = self.default('instrument_precision', 1, kwargs)
        self._initial_balance = self.default('initial_balance', 1, kwargs)
        self._price_column = self.default('price_column', 'close', kwargs)
        self._pretransform = self.default('pretransform', True, kwargs)
        self._stop_loss_percent = self.default('stop_loss_percent', 30, kwargs)
        self.leverage = self.default('leverage', 1, kwargs)
        self.n_env = self.default('n_env', 1, kwargs)
        self.data_frame = self.default('data_frame', data_frame)
        self._current_step = 0
        model = self.default('slippage_model', 'uniform', kwargs)
        self._slippage_model = slippage.get(model) if isinstance(model, str) else model()
        self.last_direction = 0
        self.total_margin = 0
        self.total_profit = 0
        self.profit = 0
        self.hold_profit = 0
        self.total_position = 0
        self.profit_percent = 0
        self.total_profit_percent = 0
        self.entry_price = 0
        self.last_price = 0

    @property
    def window_size(self) -> int:
        """The window size of observations."""
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int):
        self._window_size = window_size

        if isinstance(self.data_frame, pd.DataFrame) and self._pretransform:
            self.transform_data_frame()

    @property
    def data_frame(self) -> pd.DataFrame:
        """The underlying data model backing the price and volume simulation."""
        return getattr(self, '_data_frame', None)

    @data_frame.setter
    def data_frame(self, data_frame: pd.DataFrame):
        if not isinstance(data_frame, pd.DataFrame):
            self._data_frame = data_frame
            self._price_history = None
            return

        self._data_frame = data_frame
        self._pre_transformed_data = data_frame.copy()
        self._price_history = self._pre_transformed_data[self._price_column]
        self._pre_transformed_columns = self._pre_transformed_data.columns

        if self._pretransform:
            self.transform_data_frame()

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline=FeaturePipeline):
        self._feature_pipeline = feature_pipeline

        if isinstance(self.data_frame, pd.DataFrame) and self._pretransform:
            self.transform_data_frame()

        self.window_size_reserve_data_frame = self._data_frame.iloc[:self.window_size]
        self._data_frame = self._data_frame.iloc[self.window_size:]
        return self._feature_pipeline

    @property
    def initial_balance(self) -> float:
        return self._initial_balance

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def portfolio(self) -> Dict[str, float]:
        return self._portfolio

    @property
    def trades(self) -> pd.DataFrame:
        return self._trades

    @property
    def performance(self) -> pd.DataFrame:
        return self._performance

    @property
    def observation_columns(self) -> List[str]:
        if self._data_frame is None:
            return None

        data_frame = self._data_frame.iloc[0:10]

        if self._feature_pipeline is not None:
            data_frame = self._feature_pipeline.transform(data_frame)
            data_frame = data_frame.drop(['open', 'high', 'low'], axis=1)

        return data_frame.select_dtypes(include=[np.float, np.number]).columns

    @property
    def has_next_observation(self) -> bool:
        return self._current_step < len(self._data_frame) - 1

    @property
    def reach_to_stop_loss(self) -> bool:
        """Calculate the percentage change in net worth since the last reset.

        Returns:
            The percentage change in net worth since the last reset.
        """
        if self.profit_percent <= -self._stop_loss_percent:
            print('Reach To Stop Loss !!!')
            print('Profit Percent = ', round(self.profit_percent, 2), ' %')
            print('')
            # TODO Execute Trade = - total_position
            return True
        return False

    # @property
    # def observation_space(self) -> Box:
    #     """The final shape of the observations generated by the exchange, after any feature transformations."""
    #     n_features = len(self.observation_columns)
    #
    #     low = np.tile(self._min_trade_price, n_features)
    #     high = np.tile(self._max_trade_price, n_features)
    #     # print('low1  ', low.shape)
    #
    #     low = np.tile(low, self.n_env).reshape((self.n_env, n_features))
    #     high = np.tile(high, self.n_env).reshape((self.n_env, n_features))
    #     # print('low2  ', low.shape)
    #
    #     if self._window_size > 1:
    #         low = np.tile(low, self._window_size).reshape((self.n_env, self._window_size, n_features))
    #         high = np.tile(high, self._window_size).reshape((self.n_env, self._window_size, n_features))
    #         # print('low3  ', low.shape)
    #
    #     return Box(low=low, high=high, dtype=self._dtype)

    def _next_observation(self) -> pd.DataFrame:
        lower_range = max(self._current_step - self._window_size, 0)
        upper_range = min(self._current_step, len(self._data_frame))

        obs = self._data_frame.iloc[lower_range:upper_range]

        if not self._pretransform and self._feature_pipeline is not None:
            obs = self._feature_pipeline.transform(obs)


        if len(obs) < self._window_size:
            # padding = np.zeros((self._window_size - len(obs), len(self.observation_columns)))
            # padding = pd.DataFrame(padding, columns=self.observation_columns)
            padding = self.window_size_reserve_data_frame.iloc[len(obs):]
            obs = pd.concat([padding, obs], ignore_index=True, sort=False)

        obs = obs.select_dtypes(include='number')
        obs = obs.drop(['open', 'high', 'low'], axis=1)
        self._current_step += 1
        return obs

    def transform_data_frame(self) -> bool:
        if self._feature_pipeline is not None:
            self._data_frame = self._feature_pipeline.transform(self._pre_transformed_data)
            # self._data_frame = self._data_frame.drop(['open', 'close', 'high', 'low'], axis=1)

    def current_price(self, symbol: str) -> float:
        if self._price_history is not None:
            return float(self._price_history.iloc[self._current_step])
        return np.inf

    def _is_valid_trade(self, trade: Trade) -> bool:
        add_margin = trade.amount / (trade.price * self.leverage)
        commission = self.leverage * self._commission_percent / 100
        if self.entry_price != 0:
            plm = self.leverage * self.last_direction * ((trade.price / self.entry_price) - 1)
        else:
            plm = 0
        if self.total_position == 0 and self.balance < add_margin:
            print('Trade is not valid')
            return False
        if trade.is_long:
            if self.total_position > 0 and self.balance < add_margin:
                print('Trade is not valid')
                return False
            if self.total_position < 0:
                if trade.amount > abs(self.total_position):
                    balance = self._balance + self.total_margin * (1 - commission) * (1 + plm)
                    total_position = self.total_position + trade.amount
                    balance = balance - abs(total_position) / (trade.price * self.leverage)
                    total_margin = (1 - commission) * (abs(total_position) / (trade.price * self.leverage))
                    if balance < total_margin:
                        print('Trade is not valid')
                        return False
        elif trade.is_short:
            if self.total_position < 0 and self.balance < add_margin:
                print('Trade is not valid')
                return False
            if self.total_position > 0:
                if trade.amount > abs(self.total_position):
                    balance = self._balance + self.total_margin * (1 - commission) * (1 + plm)
                    total_position = self.total_position - trade.amount
                    balance = balance - abs(total_position) / (trade.price * self.leverage)
                    total_margin = (1 - commission) * (abs(total_position) / (trade.price * self.leverage))
                    if balance < total_margin:
                        print('Trade is not valid')
                        return False

        elif trade.is_hold:
            # print('Position ======>  Hold')
            if self.entry_price != 0:
                plm = self.leverage * self.last_direction * ((trade.price / self.entry_price) - 1)
            else:
                plm = 0
            hold_profit = (- commission * add_margin) + (self.total_margin * plm)
            if self.total_position >= 0:
                if trade.price > self.entry_price:
                    self.hold_profit = - hold_profit
                else:
                    self.hold_profit = hold_profit
            else:
                if trade.price < self.entry_price:
                    self.hold_profit = - hold_profit
                else:
                    self.hold_profit = hold_profit

            # print('Entry Price ===> ', round(self.entry_price, 2), 'USDT')
            # print('Total Position > ', round(self.total_position), 'USDT')
            # print('Price =========> ', round(trade.price, 2), 'USDT')
            # print('HOLD Profit ===> ', round(self.hold_profit, 7), 'XBT')
        return trade.amount >= self._min_trade_amount and trade.amount <= self._max_trade_amount

    def _make_trade(self, trade: Trade):
        # print('Step ==========> ', trade.step)
        if self.entry_price != 0:
            plm = self.leverage * self.last_direction * ((trade.price / self.entry_price) - 1)
        else:
            plm = 0
        self.profit_percent = plm * 100
        if self.last_price != 0:
            plm2 = self.leverage * self.last_direction * ((trade.price / self.last_price) - 1)
        else:
            plm2 = 0
        commission = self.leverage * self._commission_percent / 100
        add_margin = trade.amount / (trade.price * self.leverage)
        margin = (1 - commission) * add_margin

        if not trade.is_hold:
            self._trades = self._trades.append({
                'step': trade.step,
                'symbol': trade.symbol,
                'type': trade.trade_type,
                'amount': trade.amount,
                'price': trade.price
            }, ignore_index=True)

        if trade.is_long:
            # print('Position ======>  LONG  ▲ ')
            if self.total_position >= 0:
                self.profit = - commission * add_margin
                self._balance -= add_margin
                self.entry_price = ((self.entry_price * abs(self.total_position)) + (trade.price * trade.amount)) / (
                            abs(self.total_position) + trade.amount)
                self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) + trade.amount
                self.total_position += trade.amount
                self.total_margin = self.total_margin * (plm2 + 1) + margin
            else:
                self.profit = (- commission * add_margin) + (self.total_margin * plm)
                if trade.amount < abs(self.total_position):
                    self._balance += self.total_margin * (1 - commission) * (
                                trade.amount / abs(self.total_position)) * (1 + plm)
                    self.total_margin *= (1 - commission) * (1 - (trade.amount / abs(self.total_position))) * (1 + plm2)
                    self.total_position += trade.amount
                    self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) + trade.amount
                else:
                    self._balance += self.total_margin * (1 - commission) * (1 + plm)
                    self.total_position += trade.amount
                    self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) + trade.amount
                    self._balance -= abs(self.total_position) / (trade.price * self.leverage)
                    self.total_margin = (1 - commission) * (abs(self.total_position) / (trade.price * self.leverage))

        elif trade.is_short:
            # print('Position ======>  SHORT ▼ ')
            if self.total_position <= 0:
                self.profit = - commission * add_margin
                self._balance -= add_margin
                self.entry_price = ((self.entry_price * abs(self.total_position)) + (trade.price * trade.amount)) / (
                            abs(self.total_position) + trade.amount)
                self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) - trade.amount
                self.total_position += -trade.amount
                self.total_margin = self.total_margin * (plm2 + 1) + margin
            else:
                self.profit = (- commission * add_margin) + (self.total_margin * plm)
                if trade.amount < abs(self.total_position):
                    self._balance += self.total_margin * (1 - commission) * (
                                trade.amount / abs(self.total_position)) * (1 + plm)
                    self.total_margin *= (1 - commission) * (1 - (trade.amount / abs(self.total_position))) * (1 + plm2)
                    self.total_position += -trade.amount
                    self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) - trade.amount
                else:
                    self._balance += self.total_margin * (1 - commission) * (1 + plm)
                    self.total_position += -trade.amount
                    self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) - trade.amount
                    self._balance -= abs(self.total_position) / (trade.price * self.leverage)
                    self.total_margin = (1 - commission) * (abs(self.total_position) / (trade.price * self.leverage))

        self.total_profit = self._balance + self.total_margin - self.initial_balance
        self.total_profit_percent = ((self._balance / self.initial_balance) - 1) * 100
        # print('Position ======> ', round(trade.amount), 'USDT')
        # print('Total Position > ', round(self.total_position), 'USDT')
        # print('Entry Price ===> ', round(self.entry_price, 2), 'USDT')
        # print('Price =========> ', round(trade.price, 2), 'USDT')
        # print('PLM ===========> ', round(plm, 5), 'x')
        # print('PLM 2 =========> ', round(plm2, 5), 'x')
        # print('Add Margin ====> ', round(add_margin, 7), 'XBT')
        # print('Margin ========> ', round(margin, 7), 'XBT')
        # print('Total Margin ==> ', round(self.total_margin, 7), 'XBT')
        # print('Profit ========> ', round(self.profit, 7), 'XBT')
        # print('Profit Percent > ', round(self.profit_percent, 2), '%')
        # print('Total Profit ==> ', round(self.total_profit, 7), 'XBT')
        # print('Total Profit % > ', round(self.total_profit_percent, 2), '%')
        # print('Balance =======> ', round(self._balance, 5), 'XBT')
        if self.total_position == 0:
            self.total_margin = 0
            self.entry_price = 0
            self.last_price = 0
        self.last_direction = np.sign(self.total_position)
        self.last_price = trade.price

    def _update_account(self, trade: Trade):
        if self._is_valid_trade(trade):
            self._make_trade(trade)

        self._portfolio[self._base_instrument] = self.balance

        self._performance = self._performance.append({
            'step': self._current_step,
            'net_worth': self.net_worth,
            'balance': self.balance,
            'position_Type': trade.trade_type,
            'total_profit': self.total_profit,
            # 'total_position': self.total_position,
        }, ignore_index=True)

    def execute_trade(self, trade: Trade) -> Trade:
        current_price = self.current_price(symbol=trade.symbol)
        # commission = self._commission_percent / 100
        filled_trade = trade.copy()
        if filled_trade.is_hold or not self._is_valid_trade(filled_trade):
            filled_trade.amount = 0

        # if filled_trade.is_long:
        #     price_adjustment = 1
        #     filled_trade.price = round(current_price * price_adjustment, self._base_precision)
        #     filled_trade.amount = round((filled_trade.price * filled_trade.amount) / filled_trade.price,
        #                                 self._instrument_precision)
        # elif filled_trade.is_short:
        #     price_adjustment = 1
        #     filled_trade.price = round(current_price * price_adjustment, self._base_precision)
        #     filled_trade.amount = round(filled_trade.amount, self._instrument_precision)
        #
        # if not filled_trade.is_hold:
        #     filled_trade = self._slippage_model.fill_order(filled_trade, current_price)

        self._update_account(filled_trade)

        return filled_trade

    def reset(self):
        super().reset()

        self._current_step = 0
        self._balance = self.initial_balance
        self._portfolio = {self.base_instrument: self.balance}
        self._trades = pd.DataFrame([], columns=['step', 'symbol', 'type', 'amount', 'price'])
        self._performance = pd.DataFrame([], columns=['step', 'balance', 'net_worth'])
        self.last_direction = 0
        self.total_margin = 0
        self.total_profit = 0
        self.profit = 0
        self.hold_profit = 0
        self.total_position = 0
        self.profit_percent = 0
        self.total_profit_percent = 0
        self.entry_price = 0
        self.last_price = 0
