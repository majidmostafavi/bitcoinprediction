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


class AdvancedProfit(RewardScheme):
    """A reward scheme that rewards the agent for profitable trades and prioritizes trading over not trading.

    This scheme supports simple action schemes that trade a single position in a single instrument at a time.
    """
    def reset(self):
        """Necessary to reset the last purchase price and state of open positions."""

        self._is_holding_instrument = False
        self.last_total_profit = 0

    def get_reward(self, current_step: int, action: int, trade: Trade, total_position: float, total_profit: float, total_profit_percent: float, profit_percent: float, hold_profit: float, profit: float) -> float:

        delta = total_profit - self.last_total_profit
        if delta > 0:
            self.last_total_profit = total_profit
            return 2**np.log10(delta)
        else:
            return -1
