import gym
import torch
from pathlib import Path
from stable_baselines3 import PPO, SAC, DQN
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

from src.sb3_extentions.RPSAC import RPSAC
from src.sb3_extentions.RPPPO import RPPPO

from src.sb3_extentions.callbacks import SaveModelCallback, SaveDataCallback, ShowTrajectoryCallback, RenderCallback
from src.sb3_extentions.rp_subproc import RPSubprocVecEnv
from src.envs.random_rooms.rooms_args import get_random_rooms_arg_parser
from src.envs.rooms_render import show_rooms_traj



ENV_KW = {
    # mujoco
    'hopper':{
    'max_episode_steps':1000,
    'reward_threshold': 3800.0,
    },
    'ant':{
    'max_episode_steps': 1000,
    'reward_threshold': 6000.0
    },
    'half_cheetah':{
    'max_episode_steps':1000,
    'reward_threshold':4800.0
    },
    'pendulum':{
    'max_episode_steps':1000,
    'reward_threshold':950.0
    },
    'double_pendulum':{
    'max_episode_steps':1000,
    'reward_threshold':9100.0
    },

    # atari
    'pong': {'max_episode_steps': 10000}
}
MUJOCO_ENVS = ['hopper', 'ant', 'half_cheetah', 'pendulum', 'double_pendulum']
ATARI_ENVS = ['pong']
CNN_POLICY_ENVS = ['random_rooms', 'atari_wrapped_pong']

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
    print("args:\n", arg_dict)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Add device to args so it can propogate to RP
    print("device: ", args.device)
    if args.env_name == 'random_rooms':
        env = RPSubprocVecEnv([lambda: gym.make('random_rooms-v1',
                                          cmd_args=args)
                         for _ in range(args.n_envs)], RP_args = args)
    elif args.env_name == 'atari_wrapped_pong':
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecFrameStack
        env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
        # env = VecFrameStack(env, n_stack=4)

    else:
        # Hagay: try rp_env
        from gym.envs.registration import register
        env_kwargs = ENV_KW[args.env_name]
        env_kwargs['kwargs'] = {'args': None}
        if args.env_name in MUJOCO_ENVS:
            entry_point='src.envs.RP_env:get_rp_env' 
        elif args.env_name in ATARI_ENVS:
            entry_point='src.envs.RP_atari_env:get_rp_env' 
        register(
        id='rp_env-v0',
        entry_point=entry_point,
        **env_kwargs
        )
        create_env = lambda: gym.make('rp_env-v0', args=args)
        env = create_env()
        # env = gym.make('Pong-v0')
        env.seed(args.env_seed)

    print(env)
    print(env.__dict__)

    # env.render()
    print(env.observation_space)
    print(env.observation_space.sample().shape)

    if args.env_name in CNN_POLICY_ENVS:
        policy = 'CnnPolicy' 
        policy_kwargs = dict(features_extractor_class=CustomCnn)
    else:
        policy = 'MlpPolicy'
        policy_kwargs = {}

    if args.algo_name == 'PPO':
        model = RPPPO(policy, env, policy_kwargs=policy_kwargs, verbose=1, n_steps=args.n_steps,
                tensorboard_log=args.log_dir)

    elif args.algo_name == 'SAC':
        model = SAC(policy, env, verbose=2,
                tensorboard_log=args.log_dir)

    elif args.algo_name == 'DQN':
        model = DQN(policy, env, verbose=2,
                tensorboard_log=args.log_dir)

    elif args.algo_name == 'RPSAC':
        model = RPSAC(policy, env, verbose=2,
                tensorboard_log=args.log_dir)
        
    ###############
    # < Callbacks #
    ###############
    timesteps_interval = args.total_timesteps // args.n_data
    save_model_callback = SaveModelCallback(run_name=args.run_name)
    info_keys = []
    if args.env_name == 'random_rooms':
        info_keys.append('dijk_r')
    save_data_callback = SaveDataCallback(run_name=args.run_name, data_size=args.data_size, info_keys=info_keys, save_images=args.save_images)
    callback_list = [save_model_callback, save_data_callback]
    if args.show_trajectory:
        callback_list.append(ShowTrajectoryCallback(show_rooms_traj))
    if args.render:
        callback_list.append(RenderCallback(create_env))    
    save_callbacks = CallbackList(callback_list)
    event_callback = EveryNTimesteps(n_steps=timesteps_interval, callback=save_callbacks)
    
    callbacks = [event_callback]

    if args.wandb:
        from src.sb3_extentions.callbacks import WandbCallback
        callbacks.append(WandbCallback(args.wandb))
    
    ###############
    # Callbacks > #
    ###############

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

    # Learn model
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)


if __name__ == '__main__':
    random_rooms_parser = get_random_rooms_arg_parser()
    parser = argparse.ArgumentParser(parents=[random_rooms_parser])
    
    parser.add_argument('--env_name', required=True, type=str)
    parser.add_argument('--algo_name', required=True, type=str)

    
    
    parser.add_argument('--run_name', required=True, type=str)
    parser.add_argument('--data_size', default=0, type=int)
    parser.add_argument('--save_images', default=False, action='store_true', help="save rander images when collecting data")
    
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

    # rp online training args
    parser.add_argument('--online_training', default=False, help='use Reward Predictor, RP trains online', action='store_true')
    parser.add_argument('--user_reward', default=False, help='Get user input as reward', action='store_true')
    parser.add_argument('--max_user_reward', default=10, help='Range of user reward (0, max_user_reward)', type=int)

    parser.add_argument('--perfect_user', default=False, help='simulate user_Reward with original env reward', action='store_true')

    parser.add_argument('--win_size', default=10, type=int)
    parser.add_argument('--train_interval', default=20, type=int)
    parser.add_argument('--replay_size', default=100, type=int)
    parser.add_argument('--var_threshold', default=0.01, type=float)
    parser.add_argument('--rp_batch_size', default=16, type=int, help='windows batch size for RP training')
    
    # TODO: maybe replace this using env kwargs
    parser.add_argument('--env_max_r', default=0, type=int, help='if not 0, use for RP normaliztion')

    # Hopper Args
    parser.add_argument('--reward_distance', default=0.23, type=float) #Default should be 0.23
    parser.add_argument('--sparse_reward_value', default=1.0, type=float) #Not sure abour default
    parser.add_argument('--use_sparse_reward', default=None, choices=['accumulative', 'distance', 'window'])
    parser.add_argument('--accumulative_threshold', default=100.0,  type=float)
    parser.add_argument('--predict_with_action', default=False, action='store_true')
    # wandb
    parser.add_argument('--wandb', default=None, type=str, help='project name for W&B. Default: Wandb not active')
    parser.add_argument('--wandb_group', default=None, type=str, help='group name for W&B')
    parser.add_argument('--wandb_name', default=None, type=str, help='run name for W&B')
    ###
    parsed_args = parser.parse_args()

    run(parsed_args)
