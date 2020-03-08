from django.core.management.base import BaseCommand
from bitstealer.models import *
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.indicators import SimpleMovingAverage
from tensortrade.features.indicators import TAlibIndicator
from tensortrade.environments import TradingEnvironment
from tensortrade.features import FeaturePipeline
from tensortrade.actions import DiscreteActions
from tensortrade.rewards import SimpleProfit
from tensortrade.strategies import StableBaselinesTradingStrategy
from stable_baselines import PPO2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


candles = BitmxMinCandle.objects.all().order_by('id')[:1000]
df = pd.DataFrame(list(candles.values()))
# df = pd.read_csv('./data/Coinbase_BTCUSD_1h.csv')
exchange = SimulatedExchange(data_frame=df, base_instrument='USD', window_size=5, should_pretransform_obs=True)
# -------------------------- Feature Pipelines ------------------------#
price_columns = ["open", "high", "low", "close"]
volume_column = ["volume"]
normalized_price = MinMaxNormalizer(price_columns)
normalized_volume = MinMaxNormalizer(volume_column)
sma = SimpleMovingAverage(columns=price_columns, window_size=50)
indicators = TAlibIndicator(indicators=["EMA", "RSI", "CCI", "Stochastic", "MACD"], lows=[30, -100, 20], highs=[70, 100, 80])
difference_all = FractionalDifference(difference_order=0.6)
feature_pipeline = FeaturePipeline(steps=[normalized_price, sma, difference_all, normalized_volume, indicators])
exchange.feature_pipeline = feature_pipeline
# -------------------------- Action Schemes ------------------------#
action_scheme = DiscreteActions(n_actions=20, instrument='BTC/USD')
# -------------------------- Reward Schemes ------------------------#
reward_scheme = SimpleProfit()
# -------------------------- Live Exchange ------------------------#
# import ccxt
# from tensortrade.exchanges.live import CCXTExchange
# coinbase = ccxt.coinbasepro()
# exchange = CCXTExchange(exchange=coinbase, base_instrument='USD')
# -------------------------- Simulated Exchange ------------------------#
# df = pd.read_csv('./data/Coinbase_BTCUSD_1h.csv')
# exchange = SimulatedExchange(data_frame=df, base_instrument='USD',feature_pipeline=feature_pipeline)

# from tensortrade.exchanges.simulated import FBMExchange
# exchange = FBMExchange(base_instrument='BTC', timeframe='1h', feature_pipeline=feature_pipeline)
# #################### Creating an Environment ######################

environment = TradingEnvironment(exchange=exchange, action_scheme=action_scheme, reward_scheme=reward_scheme, feature_pipeline=feature_pipeline)
# #################### Learning Agents ######################
params = {"learning_rate": 1e-5, 'nminibatches': 1}
# agent = model(policy, environment, model_kwargs=params)
# #################### Training a Strategy ######################
strategy = StableBaselinesTradingStrategy(environment=environment, model=PPO2, policy='MlpLnLstmPolicy', model_kwargs=params)
# 'render.modes': ['human', 'rgb_array']
# episodes=1
# steps=100
performance = strategy.run(episodes=5, render_mode='human')
print(performance[:])
# performance.balance.plot()
performance.net_worth.plot()
plt.show()


# #################### Saving and Restoring ######################
# strategy.save_agent(path="../agents/ppo_btc_1h")
# from tensortrade.strategies import StableBaselinesTradingStrategy
# strategy = StableBaselinesTradingStrategy(environment=environment,
#                                        model=model,
#                                        policy=policy,
#                                        model_kwargs=params)
# strategy.restore_agent(path="../agents/ppo_btc/1h")

# #################### Tuning Your Strategy ######################
# from tensortrade.environments import TradingEnvironment
# from tensortrade.exchanges.simulated import FBMExchange
# exchange = FBMExchange(timeframe='1h', base_instrument='BTC', feature_pipeline=feature_pipeline)
# environment = TradingEnvironment(exchange=exchange, action_scheme=action_scheme, reward_scheme=reward_scheme)
# strategy.environment = environment
# tuned_performance = strategy.tune(episodes=10)

# #################### Strategy Evaluation ######################
# from pandas import pd
# from tensortrade.environments import TradingEnvironment
# from tensortrade.exchanges.simulated import SimulatedExchange
# df = pd.read_csv('./btc_ohlcv_1h.csv')
# exchange = SimulatedExchange(data_frame=df, base_instrument='BTC', feature_pipeline=feature_pipeline)
# environment = TradingEnvironment(exchange=exchange, action_scheme=action_scheme, reward_scheme=reward_scheme)
# strategy.environment = environment
# test_performance = strategy.run(episodes=1, testing=True)
