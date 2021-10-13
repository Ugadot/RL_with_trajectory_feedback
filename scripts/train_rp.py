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

from src.Reward_pred.reward_pred import Predictor

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

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--data_path', default='', help='Path to data saved with pickle')
    parser.add_argument(
        '--trajectory_length', default=1, help='trajectory length to learn with RP (default is 1)')
    parser.add_argument(
        '--save_dir', default=os.getcwd(), help='path to directory to save RP model (default is cwd)')
    parser.add_argument(
        '--model_name', default='', help='name of model to save')
    parser.add_argument(
        '--env_rows', default = 10, type = int, help='number of env rows (default is 10)')
    parser.add_argument(
        '--env_cols', default = 10, type = int, help='number of env cols (default is 10)')
    parser.add_argument(
        '--obs_type', default='conv', type=str, help='Should be either "conv" or "flat"', choices=['flat', 'conv'])
    parser.add_argument(
        '--factor', default = 1, type = float, help='pre-process reward factor')
    parser.add_argument(
        '--win_size', default = 1, type = int, help='predictor window size')
    parser.add_argument(
        '--batch_size', default = 16, type = int, help='batch size to train with')
    parser.add_argument(
        '--include_action', default = False, action='store_true')
    return parser.parse_args()

def main(args):

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    save_results_path = os.path.join(args.save_dir, "Train_results_{}.csv".format(time.strftime("%Y%m%d-%H%M%S")))

    if args.obs_type == 'flat':
        obs_shape = (
        obs_shape = (args.env_cols * args.env_rows * 3)
    elif args.obs_type == 'conv':
        obs_shape = (3, args.env_cols , args.env_rows)

    if args.obs_type == 'flat':
        reward_pred = Predictor(threshold=4, device=device,
                                num_of_parallel_envs=1,
                                show_trajectory=None,
                                siamese_type="linear",
                                input_fc_layer_size=3 * args.env_rows * args.env_cols,
                                trajectory_size=args.trajectory_length,
                                save_results_path=save_results_path,
                                obs_space = obs_shape)

    else:
        reward_pred = Predictor(threshold=4, device=device,
                                num_of_parallel_envs=1,
                                show_trajectory=None,
                                siamese_type="conv",
                                input_h=args.env_rows,
                                input_w=args.env_cols,
                                trajectory_size=args.trajectory_length,
                                save_results_path=save_results_path,
                                obs_space = obs_shape)

    train_from_data(reward_pred, args.data_path, int(args.trajectory_length), args.save_dir, args.model_name, device, args)


if __name__ == '__main__':
    args = get_args()
    main(args)
