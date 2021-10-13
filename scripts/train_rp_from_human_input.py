import os
import random
import numpy as np
import torch
import argparse
import time
import h5py

import sys
master_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, master_dir)

import gym
import wandb
from src.Reward_pred.reward_pred import Predictor
from src.utils.user_interface import Segment
from src.envs.random_rooms.rooms_args import get_random_rooms_arg_parser
from src.envs.wrappers.Dijkstra_rooms_wrapper import DijkRewardWrapper
import pickle


def preprocess(reward_list, done_list, factor):
    reward_list = np.array([i if i < 0 else 0 for i in reward_list ])

    reward_list = reward_list / factor
    #reward_list = reward_list * (-1)
    return reward_list



def test_predictor(predictor, obs_list, reward_list):
    obs_indices = list(range(len(obs_list)))
    random_indices = random.choices(obs_indices, k = 64)
    for index in random_indices:
        state = np.expand_dims(obs_list[index], 0)
        print ("Predictor reward is:\t\t" + str(np.asscalar(predictor.predict(state)[0])))
        print ("Real reward is:\t\t\t" + str(reward_list[index]))
        print("-------------------------------------------------------------------------")

def train_from_data(predictor, data_path, trajectory_length, save_dir, model_name, device, args):
    print("train_from_data, data_path: ", data_path)
    action_list = []
    obs_list = []
    reward_list = []
    done_list = []

    for subdir, dirs, files in os.walk(data_path):
        print("data files:" , files)
        for file in files:
            filepath = subdir + os.sep + file
            with h5py.File(filepath, "r") as f:
                action_list.append(f['actions'][()])
                obs_list.append(f['states'][()])
                reward_list.append(f['rewards'][()])
                done_list.append(f['dones'][()])
            # print("data shape: action: {} obs: {} reward: {} done: {}".format(action_list[-1].shape, obs_list[-1].shape,
            #                                                                   reward_list[-1].shape, done_list[-1].shape))
    action_list = np.concatenate(action_list)
    obs_list = np.concatenate(obs_list)
    reward_list = np.concatenate(reward_list)
    done_list = np.concatenate(done_list)
    reward_list = preprocess(reward_list, done_list, args.factor)
    
    assert(obs_list.shape[0] == action_list.shape[0])

    if (args.include_action):
        obs_list =  np.concatenate((obs_list, action_list), axis=1) 

    # create windows
    window_obs = []
    acc_r = []
    for i in range(obs_list.shape[0] - args.win_size):
        if True in done_list[i: i + args.win_size - 1]:
            continue
        win = obs_list[i:i + args.win_size]
        window_obs.append(win)
        acc_r.append(np.sum(reward_list[i: i+args.win_size]))
        
    window_obs = np.array(window_obs)
    acc_r = np.array(acc_r)
    acc_r = acc_r.reshape((acc_r.shape[0],1))
    print("new data shape is: obs: {} reward: {}".format(window_obs.shape, acc_r.shape))
    
    test_predictor(predictor, obs_list, reward_list)
    
    predictor.train(window_obs, acc_r, device, save_dir, model_name, batch_size = args.batch_size)
    
    #test predictor on 1 state
    test_predictor(predictor, obs_list, reward_list)


def train_from_human_sequences(predictor, data_path, save_dir, model_name, device, test_set, args):
    print("train_from_data, data_path: ", data_path)
    action_list = []
    obs_list = []
    reward_list = []
    
    for filename in os.listdir(data_path):
       if filename.endswith(".pkl"):                
           with open(os.path.join(data_path, filename), "rb") as PKL_FILE:
              sequence = pickle.load(PKL_FILE)
              action_list.append(np.array(sequence.actions).astype(np.float32))
              obs_list.append(np.stack(sequence.obs).astype(np.float32))
              reward_list.append(np.array([sequence.user_reward]).astype(np.float32))
   
    action_list = np.expand_dims(np.stack(action_list), axis=2)
    obs_list = np.stack(obs_list)
    reward_list = np.stack(reward_list)
    assert(obs_list.shape[0] == action_list.shape[0])

    #if (args.include_action):
    #    obs_list =  np.concatenate((obs_list, action_list), axis=2) 
        
    window_obs = np.array(obs_list)
    window_actions = np.array(action_list)
    acc_r = np.array(reward_list)

    test_obs = np.array(test_set["states"])
    test_actions = np.expand_dims(np.array(test_set["actions"]), axis=1)
    test_rewards = np.expand_dims(np.array(test_set["rewards"]), axis=1)


    print("new data shape is: obs: {} reward: {}".format(window_obs.shape, acc_r.shape))
        
    predictor.train(window_obs, window_actions, acc_r, test_obs, test_actions, test_rewards, device, save_dir, model_name, batch_size = args.batch_size, num_of_epochs=args.epochs)
    

def get_args():
    random_rooms_parser = get_random_rooms_arg_parser()
    parser = argparse.ArgumentParser(parents=[random_rooms_parser])
    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument(
        '--data_path', default='', help='Path to data saved with pickle')
    parser.add_argument(
        '--save_dir', default=os.getcwd(), help='path to directory to save RP model (default is cwd)')
    parser.add_argument(
        '--model_name', default='', help='name of model to save')  
    parser.add_argument(
        '--batch_size', default = 16, type = int, help='batch size to train with')
    parser.add_argument(
        '--include_action', default = False, action='store_true')
    parser.add_argument(
        '--human_sequence', default = False, action='store_true', help="data_path is directory of human rated sequences")
    parser.add_argument(
        '--epochs', default = 6, type = int, help='Number of epochs to run over all data')

    parser = DijkRewardWrapper.get_args(parser)

    # wandb
    wandb_group = parser.add_argument_group("wandb args")
    wandb_group.add_argument('--wandb', default=None, type=str, help='Project name for W&B. Default: Wandb not active')
    wandb_group.add_argument('--wandb_group', default=None, type=str, help='Group name for W&B')
    wandb_group.add_argument('--wandb_name', default=None, type=str, help='Run name for W&B')
    return parser.parse_args()



def main(args):
    if args.seed == -1:
      args.seed = None

    args.far_from_goal = False

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
    
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    env = gym.make('random_rooms-v1', cmd_args=args)
    env = DijkRewardWrapper(env, args)

    test_set_states = []
    test_set_actions = []
    test_set_rewards = []
    for i in range(64):
        obs = env.reset()
        action = random.randint(0, 3)
        res = env.step(action)
        test_set_states.append(obs)
        test_set_actions.append(action)
        test_set_rewards.append(res[1])

    test_set = {"states": test_set_states,
                "actions": test_set_actions,
                "rewards": test_set_rewards}

    action_space = None
    if args.include_action:
        action_space = env.action_space

    reward_pred = Predictor(obs_space=env.observation_space,
                        device=device,
                        model_load_path=None,
                        show_trajectory=None,  # TODO: Is this needed?
                        save_results_path=args.save_dir,
                        wandb_project=args.wandb,
                        r_norm_factor=1,
                        action_prediction_space=action_space)
                                      
    if (args.human_sequence):
        train_from_human_sequences(reward_pred, args.data_path, args.save_dir, args.model_name, device, test_set, args)
    else:
        train_from_data(reward_pred, args.data_path, int(args.trajectory_length), args.save_dir, args.model_name, device, args)


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
