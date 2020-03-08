from django.core.management.base import BaseCommand
from bitstealer.models import *
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
# from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.exchanges.live import BitmexExchange, BitmexExchange2
from tensortrade.features.scalers import MinMaxNormalizer
# from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.indicators import TAlibIndicator, SimpleMovingAverage
from tensortrade.environments import BitmexEnvironment, TradingEnvironment
from tensortrade.features import FeaturePipeline
from tensortrade.actions import ContinuousActions, CustomeDiscreteActions, DiscreteActions, DiscreteActionsPlus
from tensortrade.rewards import SimpleProfit , RiskAdjustedReturns, BitmexProfit, AdvancedProfit
from time import gmtime, strftime, localtime
import time, datetime, pytz
import gym
import logging
import os
import glob
import shutil
import numpy as np
import json
from typing import Callable, Optional, List, NamedTuple, Dict
import mlagents.trainers
from mlagents import tf_utils
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.meta_curriculum import MetaCurriculum
from mlagents.trainers.trainer_util import load_config, TrainerFactory
from mlagents.trainers.stats import TensorboardWriter, CSVWriter, StatsReporter
from mlagents.trainers.sampler_class import SamplerManager
from mlagents.trainers.exception import SamplerException
from mlagents.trainers.subprocess_env_manager import SubprocessEnvManager
from random import randint

logo = """
                 ░▄  ▄▒                                                                                                                     
                 ▒█  █▓                                                                                                                     
             ░▓▓▓██▓▓██▓▄▄     ░▄▄░               ▄▓▓█▓                                         ░▓▓▓▓▒                                      
             ▒█████████████▓  ▒████░  ████▒     ▓█████▓▒   ████░                                 ████▒                                      
             ░████▒    █████   ░▒▒░   ████▒    ▒████▓      ████░       ░▄▄▄▄░       ░░▄▄░  ░░    ████▒    ░▄▄▄▄▄░     ░░░   ░               
              █████▓▓██████░  ▓████ ░████████▓  ▓████▓   ▒████████▒  ▓████████▓   ▓███████████▒  ████░  ▒█████████▒  ████████░              
              █████▓▓▓█████▓  ▒████  ▀████▓▀▀░   ░█████▒  ▀████▓▀▀░ ████   ▓████ ████▓░░░▒████░  ████░ ▒███▒  ░████▒ █████▓▓▓░              
             ░████▒    ▒████▒ ▒████   ▓███▒       ░█████   ████    ░███████▓▓▓▓▀ ████░    ████░  ████░ ▓███████▓▓▓▓░ ████▓                  
             ▒█████▓▓▓▓█████░ ▒████   █████▄▄▓ ▒▓██████▒   █████▄▓▓ ▓███▓▄░░▄▓▓  ▓████▓▓▓█████░  ████░ ░████▄░░░▄▓░  ▓███▓                  
             ▒███████████▓▀   ▒████   ▒██████▓  █████▀     ▓██████▒  ░▓███████▓░  ░▓█████▀████▒  ████▒   ▀▓███████▀  ▓███▓                  
                 ▒█░ █▓                          ░                                                                                          
                  ░  ░░                                                                                                                     
        """

env_args = {
    "trainer_config_path":"config/sac_config.yaml",
}




class RunOptions(NamedTuple):
    trainer_config = load_config(env_args["trainer_config_path"])
    debug = False
    seed = -1
    env_path = None
    run_id = 'SAC'
    load_model = True
    train_model = True
    save_freq = 10
    keep_checkpoints = 5
    base_port = 5005
    num_envs = 1
    curriculum_config = None
    lesson = 0
    no_graphics = False
    multi_gpu = False
    sampler_config = None
    env_args = None
    cpu = True
    width = 84
    height = 84
    quality_level = 5
    time_scale = 20
    target_frame_rate = -1
    n_steps = 100
    env_id = 'MountainCar-v0'


def run_training(run_seed: int, options: RunOptions) -> None:
    """
    Launches training session.
    :param options: parsed command line arguments
    :param run_seed: Random seed used for training.
    :param run_options: Command line arguments for training.
    """
    model_path = f"./models/{options.run_id}"
    summaries_dir = "./summaries"
    port = options.base_port
    # Configure CSV, Tensorboard Writers and StatsReporter
    # We assume reward and episode length are needed in the CSV.
    csv_writer = CSVWriter(
        summaries_dir,
        required_fields=["Environment/Cumulative Reward", "Environment/Episode Length"],
    )
    tb_writer = TensorboardWriter(summaries_dir)
    StatsReporter.add_writer(tb_writer)
    StatsReporter.add_writer(csv_writer)

    if options.env_path is None:
        port = 5004  # This is the in Editor Training Port
    env_factory = create_environment_factory(
        options.env_path,
        options.no_graphics,
        run_seed,
        port,
        options.env_args,
        options.env_id,
        options.n_steps
    )
    env_manager = SubprocessEnvManager(env_factory=env_factory, n_env=options.num_envs)
    maybe_meta_curriculum = try_create_meta_curriculum(
        options.curriculum_config, env_manager, options.lesson
    )
    sampler_manager, resampling_interval = create_sampler_manager(
        options.sampler_config, run_seed
    )
    trainer_factory = TrainerFactory(
        options.trainer_config,
        summaries_dir,
        options.run_id,
        model_path,
        options.keep_checkpoints,
        options.train_model,
        options.load_model,
        run_seed,
        maybe_meta_curriculum,
        options.multi_gpu,

    )

    # Create controller and begin training.
    tc = TrainerController(
        trainer_factory=trainer_factory,
        model_path=model_path,
        summaries_dir=summaries_dir,
        run_id=options.run_id,
        save_freq=options.save_freq,
        meta_curriculum=maybe_meta_curriculum,
        train=options.train_model,
        training_seed=run_seed,
        sampler_manager=sampler_manager,
        resampling_interval=resampling_interval,
        n_steps=options.n_steps
    )
    # Begin training
    try:
        tc.start_learning(env_manager)
    finally:
        env_manager.close()


def create_environment_factory(
        env_path: Optional[str],
        no_graphics: bool,
        seed: Optional[int],
        start_port: int,
        env_args: Optional[List[str]],
        env_id: str,
        n_steps:int
) -> Callable:
    # ) -> Callable[[int, List[SideChannel]], BaseEnv]:
    if env_path is not None:
        # Strip out executable extensions if passed
        env_path = (
            env_path.strip()
                .replace(".app", "")
                .replace(".exe", "")
                .replace(".x86_64", "")
                .replace(".x86", "")
        )

    seed_count = 10000
    seed_pool = [np.random.randint(0, seed_count) for _ in range(seed_count)]

    def create_unity_environment(
            worker_id: int
    ):
        window_size = 10
        leverage = 25
        initial_balance = 100
        max_allowed_amount = 5000
        stop_loss_percent = 60
        n_splitt = 5
        env_seed = seed
        if not env_seed:
            env_seed = seed_pool[worker_id % len(seed_pool)]
        start_candle = 99
        end_candle = start_candle + n_steps + 1

        # df = pd.read_csv('./tensortrade/data/Coinbase_BTCUSD_d.csv')[start_candle:end_candle]
        # candles = BitmxMinCandle.objects.all().order_by('id')[start_candle:end_candle] #221000 (100K-150K Deleted)
        candles = MinCandle.objects.all().order_by('id')[start_candle:end_candle]  # 204000
        # candles = Min3Candle.objects.all().order_by('id')[start_candle:end_candle] #68000
        # candles = Min5Candle.objects.all().order_by('id')[start_candle:end_candle] #41000
        # candles = HourlyCandle.objects.all().order_by('id')[start_candle:end_candle] #23900
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
        ema = TAlibIndicator(indicators=[['EMA', {'args': ['close'], 'params': {'timeperiod': 14}}]])
        # sma50 = TAlibIndicator(indicators=[['SMA', {'args':['close'], 'params':{'timeperiod':50}}]])
        sma100 = TAlibIndicator(indicators=[['SMA', {'args': ['close'], 'params': {'timeperiod': 100}}]])
        # sma200 = TAlibIndicator(indicators=[['SMA', {'args':['close'], 'params':{'timeperiod':200}}]])
        rsi = TAlibIndicator(indicators=[['RSI', {'args': ['close'], 'params': {'timeperiod': 14}}]])
        macd = TAlibIndicator(indicators=[['MACD', {'args': ['close'], 'params': {'fastperiod': 5, 'slowperiod': 20, 'signalperiod': 30}}]])
        stochastic = TAlibIndicator(indicators=[['STOCH', {'args': ['high', 'low', 'close'], 'params': {'fastk_period': 5, 'slowk_period': 3, 'slowd_period': 2}}]])
        cci = TAlibIndicator(indicators=[['CCI', {'args': ['high', 'low', 'close'], 'params': {'timeperiod': 20}}]])
        feature_pipeline = FeaturePipeline(steps=[normalized_price, normalized_volume, macd, stochastic])
        exchange = BitmexExchange(data_frame=df, base_instrument='USDT', window_size=window_size,
                                  initial_balance=initial_balance, commission_percent=0.60, leverage=leverage,
                                  stop_loss_percent=stop_loss_percent)
        # action_scheme = DiscreteActions(n_actions=3, instrument='BTC/USDT', max_allowed_amount=max_allowed_amount)
        action_scheme = DiscreteActionsPlus(n_actions=4, instrument='BTC/USDT', max_allowed_amount=max_allowed_amount)
        # action_scheme = CustomeDiscreteActions(n_splitt=n_splitt, instrument='BTC/USDT', max_allowed_amount=max_allowed_amount)
        reward_scheme = BitmexProfit()
        # reward_scheme = AdvancedProfit()
        environment = BitmexEnvironment(exchange=exchange, action_scheme=action_scheme, reward_scheme=reward_scheme,
                                        feature_pipeline=feature_pipeline)
        return environment

    return create_unity_environment


def try_create_meta_curriculum(
        curriculum_config: Optional[Dict], env: SubprocessEnvManager, lesson: int
) -> Optional[MetaCurriculum]:
    if curriculum_config is None:
        return None
    else:
        meta_curriculum = MetaCurriculum(curriculum_config)
        # TODO: Should be able to start learning at different lesson numbers
        # for each curriculum.
        meta_curriculum.set_all_curricula_to_lesson_num(lesson)
        return meta_curriculum


def create_sampler_manager(sampler_config, run_seed=None):
    resample_interval = None
    if sampler_config is not None:
        if "resampling-interval" in sampler_config:
            # Filter arguments that do not exist in the environment
            resample_interval = sampler_config.pop("resampling-interval")
            if (resample_interval <= 0) or (not isinstance(resample_interval, int)):
                raise SamplerException(
                    "Specified resampling-interval is not valid. Please provide"
                    " a positive integer value for resampling-interval"
                )

        else:
            raise SamplerException(
                "Resampling interval was not specified in the sampler file."
                " Please specify it with the 'resampling-interval' key in the sampler config file."
            )

    sampler_manager = SamplerManager(sampler_config, run_seed)
    return sampler_manager, resample_interval


def run_cli(options: RunOptions) -> None:
    # print(logo)

    trainer_logger = logging.getLogger("mlagents.trainers")
    env_logger = logging.getLogger("mlagents_envs")

    if options.debug:
        trainer_logger.setLevel("DEBUG")
        env_logger.setLevel("DEBUG")
    else:
        # disable noisy warnings from tensorflow.
        tf_utils.set_warnings_enabled(False)

    trainer_logger.debug("Configuration for this run:")
    trainer_logger.debug(json.dumps(options._asdict(), indent=4))

    run_seed = options.seed
    if options.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if options.seed == -1:
        run_seed = randint(0,2)

    run_training(run_seed, options)


class Command(BaseCommand):
    help = 'TensorTrade'

    def handle(self, *args, **kwargs):
        print('Start Time is : ', datetime.datetime.now(pytz.timezone('Asia/Tehran')).strftime("%d %b %Y %H:%M:%S Tehran"))
        # print('Start Time is : ', strftime("%Y-%m-%d %H:%M:%S", localtime()))
        start_time = int(time.time())

        # arguments = parse_command_line()
        # print(RunOptions.from_argparse())
        run_cli(RunOptions())

        print('Elapsed Time is : ', round((int(time.time())-start_time)/60, 2), 'Minutes')

        results = pd.read_csv('./results.csv', index_col=0)
        results = results.drop(["step","net_worth","position_Type","total_profit"], axis=1)
        results.plot(label='Balance', color='green')
        plt.xlabel('Steps ('+str(round(len(results)))+')')
        plt.ylabel('Balance (XBT)')
        plt.legend()
        plt.show()



