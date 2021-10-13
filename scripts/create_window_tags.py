import os
import random
import numpy as np
import torch
import argparse
import time
import h5py
import wandb
import sys
from multiprocessing import Process, Queue

master_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print (master_dir)
sys.path.insert(1, master_dir)

from src.utils.user_interface import PrefInterface, Segment


def test_predictor(predictor, obs_list, reward_list):
    obs_indices = list(range(len(obs_list)))
    random_indices = random.choices(obs_indices, k = 64)
    for index in random_indices:
        state = np.expand_dims(obs_list[index], 0)
        print ("Predictor reward is:\t\t" + str(np.asscalar(predictor.predict(state)[0])))
        print ("Real reward is:\t\t\t" + str(reward_list[index]))
        print("-------------------------------------------------------------------------")

def collect_all_unlabeld_data(data_path, win_size, save_dir, args):
    print("train_from_data, data_path: ", data_path)
    action_list = []
    obs_list = []
    reward_list = []
    done_list = []
    images_list = []

    for subdir, dirs, files in os.walk(data_path):
        print("data files:" , files)
        for file in files:
            filepath = subdir + os.sep + file
            with h5py.File(filepath, "r") as f:
                action_list.append(f['actions'][()])
                obs_list.append(f['states'][()])
                reward_list.append(f['rewards'][()])
                done_list.append(f['dones'][()])
                images_list.append(f['images'][()])
    action_list = np.concatenate(action_list)
    obs_list = np.concatenate(obs_list)
    reward_list = np.concatenate(reward_list)
    done_list = np.concatenate(done_list)
    images_list = np.concatenate(images_list)
    assert(obs_list.shape[0] == action_list.shape[0])
   
   # Add action data to obs if needed
    if (args.include_action):
        obs_list =  np.concatenate((obs_list, action_list), axis=1) 
    
    # create windows
    window_obs = []
    acc_r = []
    window_images = []
    for i in range(obs_list.shape[0] - args.win_size):
        if True in done_list[i: i + args.win_size - 1]:
            continue
        win = obs_list[i:i + args.win_size]
        window_obs.append(win)
        #acc_r.append(np.sum(reward_list[i: i+args.win_size]))
        acc_r.append(reward_list[i: i+args.win_size])
        win_images = images_list[i:i + args.win_size]
        window_images.append(win_images)
        
    window_obs = np.array(window_obs)
    acc_r = np.array(acc_r)
    #acc_r = acc_r.reshape((acc_r.shape[0],1))
    window_images = np.array(window_images)
    
    
    queue_size=10
    user_queue = Queue(maxsize=queue_size)
    response_queue = Queue()
    pref_interface, pref_process = start_pref_interface(user_queue, response_queue, queue_size, 
                                                                                max_user_r=10,
                                                                                synthetic_prefs=False)
                                                                                
    # Start labeling the windows
    for i in range(len(window_obs)):
        new_segment = Segment(window_images[i], window_obs[i], acc_r[i])
        user_queue.put(new_segment)
        while (not response_queue.empty()):
             obs, user_reward, traj_mask = response_queue.get()  # block=False)
             print (user_reward)
        #need to save those
        

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--data_path', default='', help='Path to data saved with pickle')
    parser.add_argument(
        '--save_dir', default=os.getcwd(), help='path to directory to save RP model (default is cwd)')
    parser.add_argument(
        '--win_size', default = 1, type = int, help='predictor window size')
    parser.add_argument(
        '--include_action', default = False, action='store_true')
        
    return parser.parse_args()


def start_pref_interface(seg_pipe, pref_pipe, max_segs,  max_user_r, synthetic_prefs=False,
                         log_dir=None):
    def f():
        # The preference interface needs to get input from stdin. stdin is
        # automatically closed at the beginning of child processes in Python,
        # so this is a bit of a hack, but it seems to be fine.
        sys.stdin = os.fdopen(0)
        pi.run(seg_pipe=seg_pipe, pref_pipe=pref_pipe)

    # Needs to be done in the main process because does GUI setup work
    #prefs_log_dir = osp.join(log_dir, 'pref_interface')
    pi = PrefInterface(synthetic_prefs=synthetic_prefs,
                       max_segs=max_segs,
                       max_user_r=max_user_r,
                       log_dir=None)

    proc = Process(target=f, daemon=True)
    proc.start()
    return pi, proc
    
def main(args):

     save_results_path = os.path.join(args.save_dir, "Train_results_{}".format(time.strftime("%Y%m%d-%H%M%S")))
     collect_all_unlabeld_data(args.data_path, int(args.win_size), args.save_dir, args)


if __name__ == '__main__':
    args = get_args()
    main(args)
