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

import pandas as pd
import numpy as np

from tensortrade.rewards import RewardScheme
from tensortrade.trades import TradeType, Trade


class BitmexProfit(RewardScheme):
    """A reward scheme that rewards the agent for profitable trades and prioritizes trading over not trading.

    This scheme supports simple action schemes that trade a single position in a single instrument at a time.
    """
    def reset(self):
        """Necessary to reset the last purchase price and state of open positions."""

        self._is_holding_instrument = False

    def get_reward(self, current_step: int, action:int, trade: Trade, total_position: float,
                                                        total_profit_percent: float,
                                                        profit_percent: float,
                                                        hold_profit: float,
                                                        total_profit: float,
                                                        profit: float) -> float:

        """Reward -1 for not holding a position, 1 for holding a position, 2 for opening a position, and 1 + 5^(log_10(profit)) for closing a position.

        The 5^(log_10(profit)) function simply slows the growth of the reward as trades get large.
        """
        if trade.is_hold:
            return -1
        elif profit > 0:
            return 2**np.log10(profit)
        else:
            return -1
        # if trade.is_hold and self._is_holding_instrument:
        #     # print('Reward ========> ', round(hold_profit * 10, 5), '    Points')
        #     # print('')
        #     # print('-------------------------------------------------------------------------')
        #     # print('')
        #     return hold_profit
        #
        # elif (trade.is_long or trade.is_short) and trade.amount > 0:
        #
        #     if total_position != 0:
        #         self._is_holding_instrument = True
        #         # return profit_sign * (1 + (5 ** np.log10(abs(profit))))
        #         # return total_profit_percent / 100
        #         # print('Reward ========> ', round(profit * 10, 5), '    Points')
        #         # print('')
        #         # print('-------------------------------------------------------------------------')
        #         # print('')
        #         return total_profit_percent * 10
        #     else:
        #         self._is_holding_instrument = False
        #         # return profit_sign * (1 + (5 ** np.log10(abs(profit))))
        #         # return total_profit_percent / 100
        #         # print('Reward ========> ', round(profit * 10, 5), '    Points')
        #         # print('')
        #         # print('-------------------------------------------------------------------------')
        #         # print('')
        #         return total_profit_percent * 10
        #
        # # print('Reward ========> ', -1, '    Points')
        # # print('')
        # # print('-------------------------------------------------------------------------')
        # # print('')
        # return -1
