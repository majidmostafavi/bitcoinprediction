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
# limitations under the License.

import os
import gym
import json

import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List, Dict

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines import DQN

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.environments.bitmex_environment import BitmexEnvironment
from tensortrade.strategies import TradingStrategy


class BitmexTradingStrategySBL(TradingStrategy):
    """A trading strategy capable of self tuning, training, and evaluating with stable-baselines.

    Arguments:
        environments: An instance of a trading environments for the agent to trade within.
        model: The RL model to create the agent with.
            Defaults to DQN.
        policy: The RL policy to train the agent's model with.
            Defaults to 'MlpPolicy'.
        model_kwargs: Any additional keyword arguments to adjust the model.
        kwargs: Optional keyword arguments to adjust the strategy.
    """

    def __init__(self,
                 environment: BitmexEnvironment,
                 model: BaseRLModel = DQN,
                 policy: Union[str, BasePolicy] = 'MlpPolicy',
                 model_kwargs: any = {},
                 policy_kwargs: any = {},
                 n_env: int = 1,
                 **kwargs):
        self._model = model
        self._model_kwargs = model_kwargs
        self._policy_kwargs = policy_kwargs
        self._n_env = n_env

        self.environment = environment
        self._agent = self._model(policy, self._environment, **self._model_kwargs, policy_kwargs=self._policy_kwargs)

    @property
    def environment(self) -> 'BitmexEnvironment':
        """A `BitmexEnvironment` instance for the agent to trade within."""
        return self._environment

    @environment.setter
    def environment(self, environment: 'BitmexEnvironment'):
        envs = [lambda: environment for _ in range(self._n_env)]

        if self._n_env == 1:
            self._environment = DummyVecEnv(envs)
        else:
            self._environment = SubprocVecEnv(envs)

    def restore_agent(self, path: str, custom_objects: any = {}):
        """Deserialize the strategy's learning agent from a file.

        Arguments:
            path: The `str` path of the file the agent specification is stored in.
        """
        self._custom_objects = custom_objects
        self._agent = self._model.load(path, env=self._environment, custom_objects=self._custom_objects, kwargs=self._model_kwargs)

    def save_agent(self, path: str):
        """Serialize the learning agent to a file for restoring later.

        Arguments:
            path: The `str` path of the file to store the agent specification in.
        """
        self._agent.save(path)

    def tune(self, steps: int = None, episodes: int = None,
             callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def _train_callback(self, _locals, _globals):
        # performance = self._environment.performance
        #
        # if self._episode_callback and self._environment.done():
        #     self._episode_callback(performance)

        return True

    def train(self,
              steps: int = None,
              episodes: int = None,
              render_mode: str = None,
              episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        if steps is None:
            raise ValueError(
                'You must set the number of `steps` to train the strategy.')

        self._agent.learn(steps, callback=self._train_callback)

        return True

    def test(self,
             steps: int = None,
             episodes=None,
             render_mode: str = None,
             episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        if steps is None and episodes is None:
            raise ValueError(
                'You must set the number of `steps` or `episodes` to test the strategy.')

        steps_completed, episodes_completed, average_reward = 0, 0, 0
        obs, state, dones = self._environment.reset(), None, [False]
        performance = {}

        while (steps is not None and (steps == 0 or steps_completed < steps)) or (
                episodes is not None and episodes_completed < episodes):
            actions, state = self._agent.predict(obs, state=state, mask=dones)
            # actions, state = self._agent.predict(obs)
            obs, rewards, dones, info = self._environment.step(actions)

            steps_completed += 1
            average_reward -= average_reward / steps_completed
            average_reward += rewards[0] / (steps_completed + 1)

            exchange_performance = info[0].get('exchange').performance
            performance = exchange_performance if len(exchange_performance) > 0 else performance
            if render_mode is not None:
                self._environment.render(mode=render_mode)

            if dones[0]:
                if episode_callback is not None and not episode_callback(performance):
                    break

                episodes_completed += 1
                obs = self._environment.reset()

        print("Finished running strategy.")
        print("Total episodes: {} ({} timesteps).".format(episodes_completed, steps_completed))
        print("Average reward: {}.".format(average_reward))

        return performance
