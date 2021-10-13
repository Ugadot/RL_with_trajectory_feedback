# ==================================================================
# Clean run of sb3 rl algo on gym env
# ==================================================================
import gym
from stable_baselines3 import PPO, SAC
import torch
from pathlib import Path
from stable_baselines3 import PPO, SAC
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

from src.sb3_extentions.callbacks import SACWandbCallback

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
    print("args:\n", arg_dict)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Add device to args so it can propogate to RP


    env = gym.make(args.env_name)
    env.seed(args.env_seed)
    print(env)

    if args.algo_name == 'PPO':
        policy_kwargs = dict(features_extractor_class=CustomCnn)
        model = RPPPO('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1, n_steps=args.n_steps,
                tensorboard_log=args.log_dir)

    elif args.algo_name == 'SAC':
        model = SAC('MlpPolicy', env, verbose=2,
                tensorboard_log=args.log_dir)
                


    if args.wandb:
        # overload sb3 logger to use wandb
        from stable_baselines3.common import logger
        orig_record = logger.record
        def wandb_record(*args, **kwargs):
            if 'timesteps' in args[0]:
                wandb_record.step = args[1]
            if 'train' not in args[0]:
                wandb.log({args[0]: args[1]})
            orig_record(*args, **kwargs)

        wandb_record.step = 0
        logger.record = wandb_record

    callback = SACWandbCallback(args.wandb)

    # Learn model
    model.learn(total_timesteps=args.total_timesteps, callback=callback)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', required=True, type=str)
    parser.add_argument('--algo_name', required=True, type=str)

    
    
    parser.add_argument('--run_name', required=True, type=str)
    parser.add_argument('--data_size', default=0, type=int)
    parser.add_argument('--total_timesteps', default=1000000, type=int)
    parser.add_argument('--env_seed', default=-1, type=int)
    parser.add_argument('--env_size', default=15, type=int)
    parser.add_argument('--n_envs', default=2, type=int)
    parser.add_argument('--n_steps', default=2048, type=int)
    parser.add_argument('--log_dir', default=None, type=str)
    parser.add_argument('--show_trajectory', default=False, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    
    parser.add_argument('--reward_pred', default=None, type=str, help='Path to a directory containing only RP models')
    parser.add_argument('--rp_factor', default=1, type=float, help='multiply factor for predicted reward')
    parser.add_argument('--dijk_lambda', default=1, type=float, help='multiply factor for dijk reward')
    parser.add_argument('--far_from_goal', default=False, help='make agent as far from goal as possible', action='store_true')

    parser.add_argument('--n_data', default=10, type=int)
    # wandb
    parser.add_argument('--wandb', default=None, type=str, help='project name for W&B. Default: Wandb not active')
    parser.add_argument('--wandb_group', default=None, type=str, help='group name for W&B')
    parser.add_argument('--wandb_name', default=None, type=str, help='run name for W&B')

    parsed_args = parser.parse_args()
    run(parsed_args)
