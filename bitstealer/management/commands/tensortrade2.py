from django.core.management.base import BaseCommand
from bitstealer.models import *
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.exchanges.live import BitmexExchange, BitmexExchange2
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.indicators import TAlibIndicator, SimpleMovingAverage
from tensortrade.environments import BitmexEnvironment, TradingEnvironment
from tensortrade.features import FeaturePipeline
from tensortrade.actions import ContinuousActions, CustomeDiscreteActions, DiscreteActions
from tensortrade.rewards import SimpleProfit , RiskAdjustedReturns, BitmexProfit, AdvancedProfit
from tensortrade.strategies import StableBaselinesTradingStrategy, BitmexTradingStrategySBL, BitmexTradingStrategyTF
from stable_baselines import PPO2
from time import gmtime, strftime, localtime
import time, datetime, pytz


class Command(BaseCommand):
    help = 'TensorTrade'

    def handle(self, *args, **kwargs):
        print('Start Time is : ', datetime.datetime.now(pytz.timezone('Asia/Tehran')).strftime("%d %b %Y %H:%M:%S Tehran"))
        # print('Start Time is : ', strftime("%Y-%m-%d %H:%M:%S", localtime()))
        start_time = int(time.time())

        train = True
        train = False
        iteration = 500
        n_step_multiply = 2
        window_size = 50
        n_step = 16
        gamma = 0.995
        learning_rate = 2.5E-4
        lam = 0.9
        ent_coef = 0.1
        leverage = 10
        initial_balance = 100
        max_allowed_amount = 5000
        stop_loss_percent = 60
        n_splitt = 5
        end_candle = 7200
        render_mode = 'chart2'

        # df = pd.read_csv('./tensortrade/data/Coinbase_BTCUSD_d.csv')[0:end_candle]
        # candles = BitmxMinCandle.objects.all().order_by('id')[99:end_candle] #100000
        # candles = MinCandle.objects.all().order_by('id')[99:end_candle] #134000
        # candles = Min3Candle.objects.all().order_by('id')[99:end_candle] #48000
        # candles = Min5Candle.objects.all().order_by('id')[99:end_candle] #28800
        candles = HourlyCandle.objects.all().order_by('id')[0:end_candle] #22500
        df = pd.DataFrame(list(candles.values()))
        df = df.drop(['id', 'coin_marketcap', 'marketcap', 'symbol', 'date'], axis=1)
        df['open'] = df['open'].astype('float64')
        df['close'] = df['close'].astype('float64')
        df['low'] = df['low'].astype('float64')
        df['high'] = df['high'].astype('float64')
        df['volume'] = df['volume'].astype('float64')
        price_columns = ["open", "high", "low", "close"]
        volume_column = ["volume"]
        normalized_price = MinMaxNormalizer(columns=price_columns, feature_min=1E-6, feature_max=1, input_min=1E-6, input_max=1E6)
        normalized_volume = MinMaxNormalizer(columns=volume_column, feature_min=1E-9, feature_max=1, input_min=1E-9, input_max=1E9)
        # difference_all = FractionalDifference(difference_order=0.6)
        # ema = TAlibIndicator(indicators=[['EMA', {'args':['close'], 'params':{'timeperiod':14}}]])
        # sma50 = TAlibIndicator(indicators=[['SMA', {'args':['close'], 'params':{'timeperiod':50}}]])
        # sma100 = TAlibIndicator(indicators=[['SMA', {'args':['close'], 'params':{'timeperiod':100}}]])
        # sma200 = TAlibIndicator(indicators=[['SMA', {'args':['close'], 'params':{'timeperiod':200}}]])
        rsi = TAlibIndicator(indicators=[['RSI', {'args':['close'], 'params':{'timeperiod':14}}]])
        macd = TAlibIndicator(indicators=[['MACD', {'args':['close'], 'params':{'fastperiod':5,'slowperiod':20,'signalperiod':30}}]])
        stochastic = TAlibIndicator(indicators=[['STOCH', {'args':['high', 'low', 'close'], 'params':{'fastk_period':5,'slowk_period':3,'slowd_period':2}}]])
        cci = TAlibIndicator(indicators=[['CCI', {'args':['high', 'low', 'close'], 'params':{'timeperiod':20}}]])
        feature_pipeline = FeaturePipeline(steps=[normalized_volume, macd, stochastic, cci, rsi])
        exchange = BitmexExchange2(data_frame=df, base_instrument='USDT', window_size=window_size, initial_balance=initial_balance, commission_percent=0.60, leverage=leverage, stop_loss_percent=stop_loss_percent)
        # action_scheme = DiscreteActions(n_actions=3, instrument='BTC/USDT', max_allowed_amount=max_allowed_amount)
        action_scheme = CustomeDiscreteActions(n_splitt=n_splitt, instrument='BTC/USDT', max_allowed_amount=max_allowed_amount)
        reward_scheme = BitmexProfit()
        # reward_scheme = AdvancedProfit()
        environment = BitmexEnvironment(exchange=exchange, action_scheme=action_scheme, reward_scheme=reward_scheme, feature_pipeline=feature_pipeline)
        model_kwargs = {'learning_rate': learning_rate, 'nminibatches': 1, 'gamma': gamma, 'lam': lam, 'noptepochs': 4, 'n_steps': n_step, 'ent_coef': ent_coef,
                        # 'cliprange_vf': -1,
                        # 'tensorboard_log': "./tensortrade/logs/"
                        }
        # net_arch = [1024, 'lstm', dict(vf=[256, 64], pi=[64])]
        # net_arch = [1024, 'lstm', 512, 128]
        net_arch = [256, 'lstm', 512, 64]
        # net_arch = [1024, 256]
        policy_kwargs = {'net_arch': net_arch,
                         # 'feature_extraction': 'mlp',
                         'act_fun': tf.nn.relu,
                         # 'n_env': 32,
                         }
        strategy = BitmexTradingStrategySBL(environment=environment, model=PPO2, policy='MlpLstmPolicy', model_kwargs=model_kwargs, policy_kwargs=policy_kwargs)

        custom_objects = {'learning_rate': learning_rate, 'nminibatches': 1, 'gamma': gamma, 'lam': lam, 'n_steps': n_step, 'ent_coef': ent_coef}
        # custom_objects = {}
        if train:
            for i in range(iteration):
                print('*-------------- iteration =', i+1, ' --------------*')
                try:
                    strategy.restore_agent(path="./tensortrade/agents/train_2", custom_objects=custom_objects)
                    print('Agent Loaded ', datetime.datetime.now(pytz.timezone('Asia/Tehran')).strftime("%H:%M:%S"))
                    # time.sleep(3)
                except:
                    print('Loading Failed: Agent is not exist')
                    print('New Agent created ', datetime.datetime.now(pytz.timezone('Asia/Tehran')).strftime("%H:%M:%S"))
                    # time.sleep(3)
                print('Training ...')
                strategy.train(steps=round(len(df) * n_step_multiply))
                strategy.save_agent(path="./tensortrade/agents/train_2")
                print('Agent Saved ', datetime.datetime.now(pytz.timezone('Asia/Tehran')).strftime("%H:%M:%S"))
                print('Elapsed Time is : ', round((int(time.time())-start_time)/60, 2), 'Minutes')
        else:
            strategy.restore_agent(path="./tensortrade/agents/train_2")
            performance = strategy.test(steps=round(len(df) - window_size - 2), render_mode=render_mode)
            performance.balance.plot(label='Balance', color='green')
            print(performance)
            plt.xlabel('Steps ('+str(round(len(df) - window_size))+')')
            plt.ylabel('Balance (XBT)')
            plt.legend()
            plt.show()

        print('Elapsed Time is : ', round((int(time.time())-start_time)/60, 2), 'Minutes')

        # tensorboard --logdir /home/coinstealer/www/binanceweb/tensortrade/logs/
        # performances = []
        # iteration = 10
        # for i in range(iteration):
        #     print('-------------- iteration >', i, '    ', datetime.datetime.now(pytz.timezone('Asia/Tehran')).strftime("%d %b %Y %H:%M:%S Tehran"), ' --------------')
        #
        #     # performances.append(strategy.run(steps=round(len(df)*0.96), render_mode='chart2'))# steps=100 episodes=1 'render.modes': ['log', 'chart']
        #     performances.append(
        #         strategy.run(steps=round(len(df) * 0.96)))  # steps=100 episodes=1 'render.modes': ['log', 'chart']
        #     if i == 0:
        #         performances[i].balance.plot(label=i, color='blue')
        #     elif i == max(range(iteration)):
        #         performances[i].balance.plot(label=i, color='green')
        #     elif i % int(iteration / 10) == 0:
        #         performances[i].balance.plot(label=i, color=(0.9, 0.3, 0.5, i / iteration))

        # print(performances[0])


        # agent = model(policy, environment, model_kwargs=params)

        # layer1 = Dense(name='layer1', size=100, input_spec={'shape': (6,), 'type': 'float'})
        # layer2 = Dense(name='layer2', size=100, input_spec={'shape': (100,), 'type': 'float'})
        # network_spec = LayeredNetwork(name='net1', layers=[layer1, layer2], inputs_spec={'shape': (7,), 'type': 'float'})
        # agent_spec = {"type": "ppo", "discount": 0.995, "likelihood_ratio_clipping": 0.2, 'network': network_spec}

        # agent_spec = {
        #     "policy": {
        #         "network": {
        #             "type": "layered",
        #             "inputs_spec": {'shape': (7,), 'type': 'float'},
        #             "layers": [
        #                 {"type": 'dense', "size": 32, 'input_spec': {'shape': (6,), 'type': 'float'}},
        #                 {"type": 'dense', "size": 32, 'input_spec': {'shape': (6,), 'type': 'float'}},
        #                 # {"type": 'softmax'},
        #             ],
        #         }
        #     },
        #     "update": 64,
        #     "memory": {"type": Replay, "capacity": 10},
        #     "objective": "policy_gradient",
        #     "reward_estimation": {
        #         "horizon": 20
        #     }
        # }

        # agent_spec = {
        #     "policy": {
        #         "network": {
        #             "type": "auto",
        #             "size": 256,
        #             "depth": 2,
        #             # "internal_rnn": True,
        #         }
        #     },
        #     "update": 64,
        #     "memory": {"type": Replay, "capacity": 200},
        #     "objective": "policy_gradient",
        #     "reward_estimation": {
        #         "horizon": 20
        #     }
        # }