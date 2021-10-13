import os
import random
import numpy as np
import torch
import argparse
import time
import h5py
import pathlib

import sys
master_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, master_dir)
from torch.utils.tensorboard import SummaryWriter
import wandb


from src.Reward_pred.Siamese_network import Siam__conv_nn


SMOOTHING = 10
TEST = 50

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


def train_from_data(model, train_data, test_data, trajectory_length, save_dir, model_name, device, args, tensorboard_writer, wandb_usage):
    print("train_from_data")
    batch_size = args.batch_size
    train_X, train_Y = train_data
    test_X, test_Y = test_data
    assert (train_X.shape[0] == train_Y.shape[0]) or (test_X.shape[0] == test_Y.shape[0])



    epoch_len = len(train_X) // batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001)
    loss_func = torch.nn.MSELoss()
    L1_loss_func = torch.nn.L1Loss()
    acc_loss = 0
    tot_iter = 0
    for e in range(args.epochs):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_len, eta_min = 0.00001)
        permutation = torch.randperm(len(train_X))
        j = 0
        for b in (range(0, train_X.shape[0], batch_size)): 
            j += 1
            optimizer.zero_grad()
            indices = permutation[b:b+batch_size]
            batch_x, batch_y = torch.FloatTensor(train_X[indices]).to(device), torch.FloatTensor(train_Y[indices]).to(device)
            batch_o = model(batch_x)
            # print(o, y)
            loss = loss_func(batch_o, batch_y)
            
            acc_loss += loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            
            if j % SMOOTHING == 0:
                avg_loss = acc_loss / SMOOTHING
                acc_loss = 0
                print("[{}/{}][{}/{}] loss: {}".format(e, args.epochs, j, epoch_len, avg_loss))
                tensorboard_writer.add_scalar("Siamese_model_MSE_loss", avg_loss, tot_iter)
                if (wandb_usage):
                    wandb.log({"Siamese_model_MSE_loss" : avg_loss}, step = tot_iter)
                tot_iter += 1
  
            if j % TEST == 0:
                model.eval()
                test_o = model(test_X.to(device))
                test_loss = loss_func(test_o, test_Y.to(device))
                test_L1_loss = L1_loss_func(test_o, test_Y.to(device))
                print("[{}/{}][{}/{}][TEST] MSE loss: {} L1 loss: {}".format(e, args.epochs, j, epoch_len, test_loss, test_L1_loss))
                tensorboard_writer.add_scalar("test_loss", test_loss, tot_iter)
                tensorboard_writer.add_scalar("test_L1_loss", test_L1_loss, tot_iter)
                if (wandb_usage):
                    wandb.log({"test_loss" : test_loss}, step = tot_iter)
                    wandb.log({"test_L1_loss" : test_L1_loss}, step = tot_iter)
                model.train()


    return model


    
    # test_predictor(predictor, obs_list, reward_list)
    
    # predictor.train(window_obs, acc_r, device, save_dir, model_name, batch_size = args.batch_size)
    
    # #test predictor on 1 state
    # test_predictor(predictor, obs_list, reward_list)



def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--data_path', default='', help='Path to data saved with pickle')
    parser.add_argument(
        '--save_dir', default=os.getcwd(), help='path to directory to save RP model (default is cwd)')
    parser.add_argument(
        '--model_name', default='', help='name of model to save')
    # parser.add_argument(
    #     '--env_rows', default = 10, type = int, help='number of env rows (default is 10)')
    # parser.add_argument(
    #     '--env_cols', default = 10, type = int, help='number of env cols (default is 10)')
    # parser.add_argument(
    #     '--obs_type', default='conv', type=str, help='Should be either "conv" or "flat"', choices=['flat', 'conv'])
    parser.add_argument(
        '--factor', default = 1, type = float, help='pre-process reward factor')
    parser.add_argument(
        '--win_size', default = 1, type = int, help='predictor window size')
    parser.add_argument(
        '--batch_size', default = 64, type = int, help='batch size to train with')
    parser.add_argument(
        '--include_action', default = False, action='store_true')
    parser.add_argument(
        '--epochs', default = 10, type = int, help='batch size to train with')
        
    # wandb
    parser.add_argument('--wandb', default=None, type=str, help='project name for W&B. Default: Wandb not active')
    parser.add_argument('--wandb_group', default=None, type=str, help='group name for W&B')
    parser.add_argument('--wandb_name', default=None, type=str, help='run name for W&B')
    ###
    
    return parser.parse_args()


def load_rooms_data(args):
    print("load_data, data_path: ", args.data_path)
    action_list = []
    obs_list = []
    reward_list = []
    dijk_r_list = []
    done_list = []

    for subdir, dirs, files in os.walk(args.data_path):
        files = [f for f in files if '0.h5' in f  or '1.h5' in f or '1.h5' in f or '2.h5' in f or '3.h5' in f]
        print("data files:" , files)
        for file in files:
            filepath = subdir + os.sep + file
            with h5py.File(filepath, "r") as f:
                action_list.append(f['actions'][()])
                obs_list.append(f['states'][()])
                reward_list.append(f['rewards'][()])
                dijk_r_list.append(f['dijk_r'][()])
                done_list.append(f['dones'][()])
            # print("data shape: action: {} obs: {} reward: {} done: {}".format(action_list[-1].shape, obs_list[-1].shape,
            #                                                                   reward_list[-1].shape, done_list[-1].shape))
    action_list = np.concatenate(action_list)
    obs_list = np.concatenate(obs_list)
    reward_list = np.concatenate(reward_list)
    done_list = np.concatenate(done_list)
    reward_list = preprocess(reward_list, done_list, args.factor)
    dijk_r_list = np.concatenate(dijk_r_list)
    
    if (args.include_action):
        obs_list =  np.concatenate((obs_list, action_list), axis=1) 
    
    # create eval set
    obs_indices = list(range(len(obs_list)))
    random_indices = random.choices(obs_indices, k = 256)
    test_obs, test_r = [], []
    for i in random_indices:
        # create windows len 1
        test_obs.append([obs_list[i]])
        test_r.append([dijk_r_list[i]])
    test_obs = torch.FloatTensor(np.array(test_obs))
    test_r = torch.FloatTensor(np.array(test_r))
    
    # create windows
    window_obs = []
    acc_r = []
    for i in range(obs_list.shape[0] - args.win_size):
        if True in done_list[i: i + args.win_size - 1]:
            continue
        win = obs_list[i:i + args.win_size]
        window_obs.append(win)
        acc_r.append(np.sum(dijk_r_list[i: i+args.win_size]))
        
    window_obs = np.array(window_obs)
    acc_r = np.array(acc_r)
    acc_r = acc_r.reshape((acc_r.shape[0],1))
    print("new data shape is: obs: {} reward: {}".format(window_obs.shape, acc_r.shape))
    
    #window_obs = torch.FloatTensor(window_obs)
    #acc_r = torch.FloatTensor(acc_r)
    return (window_obs, acc_r), (test_obs, test_r)

def main(args):

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    print("device", device)
    save_results_path = os.path.join(args.save_dir, "Train_results_{}.csv".format(time.strftime("%Y%m%d-%H%M%S")))
    
    tensorboard_dir = pathlib.Path(os.path.join(save_results_path, "tensorBoard","RP"))
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    train_data, eval_data = load_rooms_data(args)
    # net_input.to(device)
    # out.to(device)
    in_shape = train_data[0].shape[2:]
    print("train in len: {} out len: {}".format(train_data[0].shape[0], train_data[1].shape[0]))
    print("in_shape is: ", in_shape)
    
    arg_dict = vars(args)
    if args.wandb:
        wandb_args = {'project': args.wandb}
        if args.wandb_group:
            wandb_args['group'] = args.wandb_group 
        if args.wandb_name:
            wandb_args['name'] = args.wandb_name
        wandb_args['config'] = arg_dict
        wandb.init(**wandb_args)
    
    if len(in_shape) == 3:
        siam_cnn = Siam__conv_nn(observation_shape=in_shape, outputs=1, use_bn=True).to(device)

    else:
        print("unspupported in_shape")
        return(1)

    model = train_from_data(siam_cnn, train_data, eval_data, int(args.win_size), args.save_dir, args.model_name, device, args, tensorboard_writer, args.wandb)
    model_path = os.path.join(args.save_dir, "trained_model.pth")
    torch.save(model, model_path)

if __name__ == '__main__':
    args = get_args()
    main(args)
