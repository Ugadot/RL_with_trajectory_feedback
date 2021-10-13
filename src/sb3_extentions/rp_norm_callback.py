import os
import torch
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from abc import abstractmethod
from src.data.data_generator import DataGenerator
from src.utils.user_interface import saved_data_render
from multiprocessing import Process

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process
from tqdm import tqdm
import wandb

def obs_as_tensor(obs, device):
    """
    Moves the observation to the given device.
    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return torch.as_tensor(obs).to(device)
    elif isinstance(obs, dict):
        return {key: torch.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

class RPNormCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RPNormCallback, self).__init__(verbose)

    def _init_callback(self) -> None:
        self.n_envs = self.training_env.num_envs
        self.n_steps = self.model.n_steps
        self.r_ints = np.empty((self.n_envs, self.n_steps))
        self.cur_step = 0

    def _on_step(self) -> bool:
        for inx, info in enumerate(self.locals['infos']):
            self.r_ints[inx, self.cur_step] = info['r_int']
        self.cur_step += 1
        return True

    def _on_rollout_end(self) -> None:


        # TODO: cal norm_r_ints
        rp_factor = 0.01
        norm_r_ints = self.r_ints / np.std(self.r_ints)
        norm_r_ints = norm_r_ints * rp_factor
        self.locals['rollout_buffer'].rewards += norm_r_ints.T

        #call this again because collect_rollout use this before our update
        # https: // github.com / DLR - RM / stable - baselines3 / blob / 3845
        # bf9f3209173195f90752e341bbc45a44571b / stable_baselines3 / common / on_policy_algorithm.py
        # #:~:text=rollout_buffer.compute_returns_and_advantage(last_values%3Dvalues%2C%20dones%3Ddones)
        with torch.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(self.locals['new_obs'], self.locals['self'].device)
            _, values, _ = self.locals['self'].policy.forward(obs_tensor)
        self.locals['rollout_buffer'].compute_returns_and_advantage(last_values=values,
                                                                    dones=self.locals['dones'])


        # reinitialize
        self.cur_step = 0
        self.r_ints = np.empty((self.n_envs, self.n_steps))
