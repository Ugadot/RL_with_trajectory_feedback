import os
import random
import numpy as np
import torch
import argparse
import time
import h5py
import wandb
import sys
master_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, master_dir)

from src.Reward_pred.reward_pred import Predictor

def test_predictor(predictor, obs_list, reward_list):
    obs_indices = list(range(len(obs_list)))
    random_indices = random.choices(obs_indices, k = 64)
    for index in random_indices:
        state = np.expand_dims(obs_list[index], 0)
        print ("Predictor reward is:\t\t" + str(np.asscalar(predictor.predict(state)[0])))
        print ("Real reward is:\t\t\t" + str(reward_list[index]))
        print("-------------------------------------------------------------------------")

def train_from_data(predictor, data_path, win_size, save_dir, model_name, device, wandb, args):
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
    
    assert(obs_list.shape[0] == action_list.shape[0])
    
    #Normalize obs, rewards and action
    reward_mean = np.mean(reward_list)
    reward_max_value = np.max(np.abs(reward_list))
    reward_list = (reward_list - reward_mean) / reward_max_value
    
    obs_mean = np.mean(obs_list, axis=0)   
    obs_max_value = np.max(np.abs(obs_list),axis=0)
    for index, max_val in enumerate(obs_max_value):
        if (max_val == 0):
            obs_max_value[index] = 1
    obs_list = np.divide((obs_list - obs_mean), obs_max_value)
   
    action_mean = np.mean(action_list, axis=0)   
    action_max_value = np.max(np.abs(action_list),axis=0)
    for index, max_val in enumerate(action_max_value):
        if (max_val == 0):
            action_max_value[index] = 1
    action_list = np.divide((action_list - action_mean), action_max_value)
    # Finsih normalizing
   
   
   # Add action data to obs if needed
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
    
    predictor.train(window_obs, acc_r, device, save_dir, model_name, batch_size = args.batch_size, num_of_epochs = args.epochs, wandb = wandb, win_size = win_size)
    
    #test predictor on 1 state
    test_predictor(predictor, obs_list, reward_list)

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--data_path', default='', help='Path to data saved with pickle')
    parser.add_argument(
        '--save_dir', default=os.getcwd(), help='path to directory to save RP model (default is cwd)')
    parser.add_argument(
        '--model_name', default='', help='name of model to save')
    parser.add_argument(
        '--win_size', default = 1, type = int, help='predictor window size')
    parser.add_argument(
        '--batch_size', default = 16, type = int, help='batch size to train with')
    parser.add_argument(
        '--epochs', default = 2, type = int, help='num of epochs to run')
    parser.add_argument(
        '--include_action', default = False, action='store_true')
        
    # wandb
    parser.add_argument('--wandb', default=None, type=str, help='project name for W&B. Default: Wandb not active')
    parser.add_argument('--wandb_group', default=None, type=str, help='group name for W&B')
    parser.add_argument('--wandb_name', default=None, type=str, help='run name for W&B')
    ###
    
    return parser.parse_args()

def main(args):

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    save_results_path = os.path.join(args.save_dir, "Train_results_{}".format(time.strftime("%Y%m%d-%H%M%S")))

    # Need to make this relevant for other envs. maybe create a dict with fixed sizes?
    input_size = 11
    if args.include_action:
        input_size = 14
    
    arg_dict = vars(args)
    if args.wandb:
        wandb_args = {'project': args.wandb}
        if args.wandb_group:
            wandb_args['group'] = args.wandb_group 
        if args.wandb_name:
            wandb_args['name'] = args.wandb_name
        wandb_args['config'] = arg_dict
        wandb.init(**wandb_args) 
    
    reward_pred = Predictor(threshold = 4,  # TODO: add to arguments?
                            device = device,
                            env_name = "",
                            show_trajectory = None,
                            num_of_parallel_envs = 1,
                            siamese_type = " linear",
                            input_fc_layer_size = input_size,
                            trajectory_size=args.win_size,
                            save_results_path=save_results_path)
                            #wandb_project=args.wandb)


    train_from_data(reward_pred, args.data_path, int(args.win_size), args.save_dir, args.model_name, device, args.wandb, args)


if __name__ == '__main__':
    args = get_args()
    main(args)
