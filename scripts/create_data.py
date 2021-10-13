import gym
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps

import argparse
import json
import os
import sys
import numpy as np

import wandb

master_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, master_dir)

from src.sb3_extentions.custom_cnn import CustomCnn
from src.sb3_extentions.callbacks import SaveModelCallback, SaveDataCallback, ShowTrajectoryCallback, WandbCallback
from src.sb3_extentions.rp_subproc import RPSubprocVecEnv
from src.envs.random_rooms.argparse import get_random_rooms_arg_parser
from src.envs.rooms_render import show_rooms_traj

def get_env_mean_r(env, H=1000, k=100):
    action_space = env.action_space
    print(action_space)
    for i in range(k):
        rewards = []
        env.reset()
        done = False
        while not done:
            _, reward, done, _ = env.step(action_space.sample())
            rewards.append(reward)

    mean_r = sum(rewards) / len(rewards)
    print("mean reward: ", mean_r)
    return mean_r




def run(args):
    if args.env_seed == -1:
        args.env_seed =  None

    args.upsample = 1
    
    arg_dict = vars(args)
    if args.wandb:
        wandb_args = {'project': args.wandb}
        if args.wandb_group:
            wandb_args['group'] = args.wandb_group 
        if args.wandb_name:
            wandb_args['name'] = args.wandb_name
        wandb_args['config'] = arg_dict
        wandb.init(**wandb_args)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.log_dir, "args.json"), 'wt') as f:
            json.dump(arg_dict, f, indent=4)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Add device to args so it can propogate to RP


    env = RPSubprocVecEnv([lambda: gym.make('random_rooms-v1',
                                          cmd_args=args)
                         for _ in range(args.n_envs)], RP_args = args)

    policy_kwargs = dict(features_extractor_class=CustomCnn)
    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1, n_steps=args.n_steps,
                tensorboard_log=args.log_dir)
    
    #############
    # Callbacks #
    #############
    timesteps_interval = args.total_timesteps // args.n_data
    save_model_callback = SaveModelCallback(run_name=args.run_name, verbose=1)
    save_data_callback = SaveDataCallback(run_name=args.run_name, data_size=args.data_size, info_keys=['dijk_r'], verbose=1)
    callback_list = [save_model_callback, save_data_callback]
    if args.show_trajectory:
        callback_list.append(ShowTrajectoryCallback(show_rooms_traj))

    save_callbacks = CallbackList(callback_list)
    event_callback = EveryNTimesteps(n_steps=timesteps_interval, callback=save_callbacks)
    
    # model.learn(total_timesteps=args.total_timesteps, callback=event_callback)
    
    all_callbacks = [event_callback]
    
    if args.wandb:
        #wandb_callback = EveryNTimesteps(n_steps=args.n_steps, callback=WandbCallback())
        #all_callbacks.append(wandb_callback)
        from stable_baselines3.common import logger
        orig_record = logger.record
        def wandb_record(*args, **kwargs):
            wandb.log({args[0]: args[1]})
            orig_record(*args, **kwargs)
        logger.record = wandb_record
            
    all_callbacks = CallbackList(all_callbacks)
    # Learn model
    model.learn(total_timesteps=args.total_timesteps, callback=all_callbacks) #event_callback)


if __name__ == '__main__':
    random_rooms_parser = get_random_rooms_arg_parser()
    parser = argparse.ArgumentParser(parents=[random_rooms_parser])

    parser.add_argument('--run_name', required=True, type=str)
    parser.add_argument('--data_size', default=0, type=int)
    parser.add_argument('--total_timesteps', default=1000000, type=int)
    parser.add_argument('--env_seed', default=-1, type=int)
    parser.add_argument('--env_size', default=15, type=int)
    parser.add_argument('--n_envs', default=2, type=int)
    parser.add_argument('--n_steps', default=2048, type=int)
    parser.add_argument('--log_dir', default=None, type=str)
    parser.add_argument('--show_trajectory', default=False, action='store_true')
    parser.add_argument('--reward_pred', default=None, type=str, help='Path to a directory containing only RP models')
    parser.add_argument('--rp_factor', default=1, type=float, help='multiply factor for predicted reward')
    parser.add_argument('--dijk_lambda', default=1, type=float, help='multiply factor for dijk reward')
    parser.add_argument('--far_from_goal', default=False, help='make agent as far from goal as possible', action='store_true')

    parser.add_argument('--n_data', default=10, type=int)

    # rp online training args
    parser.add_argument('--online_training', default=False, help='make agent as far from goal as possible', action='store_true')
    parser.add_argument('--win_size', default=10, type=int)
    parser.add_argument('--train_interval', default=20, type=int)
    parser.add_argument('--replay_size', default=100, type=int)
    
    # wandb
    parser.add_argument('--wandb', default=None, type=str, help='project name for W&B. Default: Wandb not active')
    parser.add_argument('--wandb_group', default=None, type=str, help='group name for W&B')
    parser.add_argument('--wandb_name', default=None, type=str, help='run name for W&B')


    parsed_args = parser.parse_args()

    run(parsed_args)
