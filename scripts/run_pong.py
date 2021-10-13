import gym
from gym.wrappers import FrameStack
import torch
from pathlib import Path
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps
from stable_baselines3.common.atari_wrappers import AtariWrapper

import argparse
import json
import os
import sys
import numpy as np
import pprint
import wandb

master_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, master_dir)

from src.sb3_extentions.custom_cnn import CustomCnn

from src.sb3_extentions.callbacks import SaveModelCallback, SaveDataCallback, ShowTrajectoryCallback, RenderCallback
from src.sb3_extentions.rp_subproc import RPSubprocVecEnv
from src.sb3_extentions.callbacks import WandbCallback, WandbVecCallback

from src.envs.wrappers.pong_dense import PongDenseWrapper
from src.envs.wrappers.rp_wrapper import RPWrapper

import src.envs
import yaml
from types import SimpleNamespace

FORCE_VEC_ENV = True  # use subproc_vec_env with n_envs = 1

def run(args):
    if args.seed == -1:
        args.seed = None

    args.upsample = 1
    arg_dict = vars(args)

    if not args.wandb:
        print("Must enter wandb project, group and run name")

    wandb_args = {'project': args.wandb, 'entity': 'rl_trajectory_feedback'}
    if args.wandb_group:
        wandb_args['group'] = args.wandb_group
    if args.wandb_name:
        wandb_args['name'] = args.wandb_name
    wandb_args['config'] = arg_dict
    wandb.init(**wandb_args)

    log_dir = os.path.join(master_dir, "data", args.wandb, args.wandb_group, args.wandb_name)

    # Add lod_dir to arguments for data saving etc.
    args.log_dir = log_dir

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, "config_yaml.yml"), 'wt') as f:
        yaml.dump(arg_dict, f, indent=4)
    print("args:")
    pprint.pprint(arg_dict, width=1)
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # Add device to args so it can propagate to RP
    print("device: ", args.device)
    
    
    os.putenv('DISPLAY', 'localhost:12.0')
    
    if args.obs_type == "conv":
      pong_env = 'Pong-v0'
    else:
      pong_env = 'Pong-ram-v0'

    if args.n_envs == 1 and not FORCE_VEC_ENV:
        print("create single env")
        env = gym.make(pong_env)
        env = PongDenseWrapper(env, args)
        if args.frame_stack != 1:
          env = FrameStack(env, args.frame_stack)
        if args.use_RP:
            env = RPWrapper(env, args)
    else:
        print("create vec env")

        def create_env():
            e = gym.make(pong_env)
            e = PongDenseWrapper(e, args)
            if args.frame_stack != 1:
              e = FrameStack(e, args.frame_stack)
            return e

        if args.use_RP:
            env = RPSubprocVecEnv([create_env for _ in range(args.n_envs)], args=args)
        else:
            env = SubprocVecEnv([create_env for _ in range(args.n_envs)])

    print("Current trained environment is:")
    print(env)

    ppo_kwargs = {}
    if hasattr(args, 'n_steps'):
        ppo_kwargs['n_steps'] = args.n_steps
    if hasattr(args, 'rl_n_epcohs'):
        ppo_kwargs['n_epochs'] = args.rl_n_epcohs
    if hasattr(args, 'rl_batch_size'):
        ppo_kwargs['batch_size'] = args.rl_batch_size

    if args.obs_type == "conv":
        policy = 'CnnPolicy'
        policy_kwargs = dict(features_extractor_class=CustomCnn)
        #model = PPO(policy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir, **ppo_kwargs)
        model = PPO(policy, env, policy_kwargs=policy_kwargs, verbose=1, **ppo_kwargs)

    else:
        #model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, **ppo_kwargs)
        model = PPO("MlpPolicy", env, verbose=1, **ppo_kwargs)
    # TODO: Maybe remove tensorboard_log?

    ###############
    # < Callbacks #
    ###############
    timesteps_interval = args.total_timesteps // args.n_data
    save_model_callback = SaveModelCallback(save_path=log_dir)
    info_keys = []
    # if args.env_name == 'random_rooms':
    save_data_callback = SaveDataCallback(save_path=log_dir, data_size=args.data_size, info_keys=info_keys,
                                          save_images=args.save_images, show_images=args.show_images)
    callback_list = [save_model_callback, save_data_callback]
    save_callbacks = CallbackList(callback_list)
    event_callback = EveryNTimesteps(n_steps=timesteps_interval, callback=save_callbacks)

    callbacks = [event_callback]

    #if args.n_envs == 1 and not FORCE_VEC_ENV:
    #    callbacks.append(WandbCallback(args.wandb, every_step=True))
    #else:
    callbacks.append(WandbVecCallback(args.n_envs, args.wandb, every_step=False))


    # rp norm callback
    if args.use_RP:
        from src.sb3_extentions.rp_norm_callback import RPNormCallback
        callbacks.append(RPNormCallback())
    ###############
    # Callbacks > #
    ###############

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

    # Learn model
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', default=None, type=str, help="Path to config yaml file instead of giving all other arguments")
    parser.add_argument('--seed', default=-1, type=int)

    # RL arguments
    RL_group = parser.add_argument_group("RL algo args")
    RL_group.add_argument('--total_timesteps', default=1000000, type=int)
    RL_group.add_argument('--n_envs', default=1, type=int, help='Numer of envs to run simlutanesoly, default=1')
    RL_group.add_argument('--n_steps', default=2048, type=int)
    
    #collect data arguments
    collect_data_group = parser.add_argument_group("Collect data args")
    collect_data_group.add_argument('--n_data', default=10, type=int)
    collect_data_group.add_argument('--data_size', default=0, type=int)
    collect_data_group.add_argument('--save_images', default=False, action='store_true', help="save render images when collecting data")   
    collect_data_group.add_argument('--show_images', default=False, action='store_true', help="show saved images")

    # Get Dijkstra arguments from wrapper class function
    parser = PongDenseWrapper.get_args(parser)

    # rp online training args
    RP_group = parser.add_argument_group("Reward Predictor args")
    
    RP_group.add_argument('--use_RP', default=False, action='store_true', help='Use RP or not use at all')
    RP_group.add_argument('--RP_path', default=None, type=str, help='Path to a directory containing only RP models - DO NOT MARK --online_training with this argument')
    RP_group.add_argument('--online_training', type=str, default=None,
                          help='creates a new Reward Predictor, RP trains online, specify type',
                          choices=[None, 'perfect_user, discrete_user', 'real_user'])
    RP_group.add_argument('--max_user_reward', default=10, help='Range of user reward (0, max_user_reward)', type=int)
    RP_group.add_argument('--win_size', default=10, type=int, help='Size of window to use for RP training, default=10')
    RP_group.add_argument('--train_interval', default=20, type=int, help='Each {trin_interval} env steps the RP is trained from replay buffer, default=20')
    RP_group.add_argument('--replay_size', default=100, type=int, help='Amount of windows to save in replay buffer for RP to train on, default=100')
    RP_group.add_argument('--var_threshold', default=0.01, type=float, help='Variance threshold of RP models to ask User for input from, default=0.01')
    RP_group.add_argument('--window_ask_p', default=0.005, type=float, help='Probability to ask the user about a window even if the onditions are not met, default=0.05')
    RP_group.add_argument('--rp_batch_size', default=2, type=int, help='Windows batch size for RP training, default=2')
    RP_group.add_argument('--rp_factor', default=0.05, type=float, help='Multiply factor for predicted reward, default=0.05')
    RP_group.add_argument('-pwa', '--predict_with_action', default=False, help='Insert action to RP Network as well', action='store_true')

    # TODO: maybe replace this using env kwargs
    RP_group.add_argument('--env_max_r', default=0, type=int, help='If not 0, use for RP normaliztion')


    # wandb
    wandb_group = parser.add_argument_group("wandb args")
    wandb_group.add_argument('--wandb', default=None, type=str, help='Project name for W&B. Default: Wandb not active')
    wandb_group.add_argument('--wandb_group', default=None, type=str, help='Group name for W&B')
    wandb_group.add_argument('--wandb_name', default=None, type=str, help='Run name for W&B')
    ###

    # Parse args and run script with either config arguments or cmd_line arguments
    parsed_args = parser.parse_args()
    if (parsed_args.config_yaml is not None):
        yaml_path = parsed_args.config_yaml
        if (not os.path.exists(yaml_path)):
            print("Given config path is illegal")
            exit(1)
        with open(yaml_path) as YAML:
            config_dict = yaml.load(YAML, Loader=yaml.FullLoader)
        yaml_args = SimpleNamespace(**config_dict)
        run(yaml_args)
    else:
        run(parsed_args)