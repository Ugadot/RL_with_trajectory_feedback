import os
import sys
import numpy as np
from multiprocessing import Queue
from numpy_ringbuffer import RingBuffer
import random
import torch
import pickle
from filelock import FileLock
import glob
from pathlib import Path

file_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(file_dir)
from src.utils.user_interface import Segment, start_pref_interface


class ReplayBuffer:
    '''
    Holds windows, rewards  # TODO: Add masks buffer?
    '''
    def __init__(self, args):
        self.replay_size = args.replay_size
        self.RP_obs_replay_buffer = []
        self.RP_rewards_replay_buffer = []
        self.RP_masks_replay_buffer = []
        self.RP_actions_replay_buffer = []

    def insert(self, obs, reward, action):
        self.RP_obs_replay_buffer.append(obs)
        self.RP_rewards_replay_buffer.append(reward)
        self.RP_actions_replay_buffer.append(action)

        # Make sure replay buffer size does not get larger than
        if len(self.RP_obs_replay_buffer) > self.replay_size:
            self.RP_obs_replay_buffer = self.RP_obs_replay_buffer[1:]
            self.RP_rewards_replay_buffer = self.RP_rewards_replay_buffer[1:]
            self.RP_actions_replay_buffer = self.RP_actions_replay_buffer[1:]

    def get_replay_size(self):
        return len(self.RP_obs_replay_buffer)

    def get_one_batch(self, batch_size, device):
        index_of_data = list(range(len(self.RP_obs_replay_buffer)))

        batch_indices = random.sample(index_of_data, batch_size)

        windows_batch_tensors = [torch.FloatTensor(self.RP_obs_replay_buffer[x]) for x in batch_indices]
        rewards_batch_tensors = [self.RP_rewards_replay_buffer[x] for x in batch_indices]
        actions_batch_tensors = [self.RP_actions_replay_buffer[x] for x in batch_indices]

        batch_tensor = torch.stack(windows_batch_tensors).to(device)
        rewards_tensor = torch.FloatTensor(rewards_batch_tensors).unsqueeze(1).to(device)
        action_tensor = torch.FloatTensor(actions_batch_tensors).to(device)

        return batch_tensor, rewards_tensor, action_tensor


class SingleEnvWindowBuffer:
    '''
    RingBuffer for single env, contains user_obs, obs, vars and rewards
    '''
    def __init__(self, args):
        self.user_reward = args.online_training == 'real_user'
        self.win_size = args.win_size

        self.model_var_thresh = args.var_threshold
        self.window_ask_p = args.window_ask_p

        self.reset_buffer()

    def insert(self, user_data, rp_obs, var, rp_reward, action):
        '''
        All input arguments is per_env data

        :param user_data: current observation after being processed to user_image
        :param rp_obs: current observation after being processed to RP suitable obs
        :param vars: current variance of RP models
        :param rp_rewards: current RP predicted reward
        :param done: current done flag
        :param action: current action
        :return:
        The function insert the data to the RingBuffers from right
        '''
        self.user_data_buffer.append(user_data)
        self.window_states_buffer.append(rp_obs)
        self.predicted_var_buffer.append(var)
        self.real_reward_buffer.append(rp_reward)
        self.actions_buffer.append(action)

        # Update the current collected window size
        self.current_window_size = min(self.current_window_size + 1, self.win_size)

    def reset_buffer(self):
        '''
        Function that resets the current buffer as a part of an environment reset
        '''
        self.user_data_buffer = RingBuffer(capacity=self.win_size, dtype=np.ndarray)
        self.window_states_buffer = RingBuffer(capacity=self.win_size, dtype=np.ndarray)
        self.predicted_var_buffer = RingBuffer(capacity=self.win_size, dtype=float)
        self.real_reward_buffer = RingBuffer(capacity=self.win_size, dtype=float)
        self.actions_buffer = RingBuffer(capacity=self.win_size, dtype=np.ndarray)

        self.current_window_size = 0

    def check_if_current_window_trainable(self, done_flag):
        '''
        Check if need to train on current window
        :param done_flag: current step done flag can also trigger returned window
        :return: either Segment ready to be added to Replay buffer or None
        '''

        nd_vars_buffer = np.asarray(self.predicted_var_buffer)

        # Compute the average variance per window (np.mean cannot take into consideration the value of ring_buffer_len)
        avg_win_var = np.sum(nd_vars_buffer) / self.current_window_size

        # self.window_ask_p is a probability of taking the window into account without the VAR being larger than threshold
        variance_condition = (avg_win_var > self.model_var_thresh) | \
                             (np.random.random(avg_win_var.shape) <= self.window_ask_p)

        # TODO: How to log this part??
        # for index, info in enumerate(infos):
        #     info['wandb_log']["RP_Average_window_var"] = avg_win_var[index]

        windows_ready = (self.current_window_size >= self.win_size) | done_flag
        add_to_train_data_condition = windows_ready & variance_condition

        # If taken then return the window and zero the current window size
        if add_to_train_data_condition:
            current_segment = Segment(np.array(self.user_data_buffer),
                                      np.array(self.window_states_buffer),
                                      np.array(self.real_reward_buffer),
                                      np.array(self.actions_buffer))
            self.current_window_size = 0
            return current_segment, avg_win_var
        else:
            return None, None


class WindowBuffer:
    '''
    This class hold a window buffer for each online played env and create training windows from them
    if the training window hold several conditions, they insert the window into the RP replay buffer.
    the replay buffer is then supplied into the RP train function.

    members:
    '''

    def __init__(self, args):
        self.online_training = args.online_training
        self.max_user_reward = args.max_user_reward
        self.noisy_user = getattr(args, 'noisy_user', 0)
        self.win_size = args.win_size
        # Replay buffers are data saved for RP training, if there are several environments the windows are split
        # and inserted differently to the replay buffer
        self.replay_buffer = ReplayBuffer(args)
        self.num_envs = args.n_envs
        self.model_var_thresh = args.var_threshold
        self.window_ask_p = args.window_ask_p
        self._inserted_windows = 0

        self._log_dir = args.log_dir
        print ("log dir is: ", self._log_dir)
        if self._log_dir:
          log_segments_path = os.path.join(self._log_dir, "trained_segments")
          os.makedirs(log_segments_path, exist_ok=True)
        # Restart a list of SingleEnvWindowBuffer
        self.window_buffers = [SingleEnvWindowBuffer(args) for i in range(self.num_envs)]

        if self.online_training == 'real_user':
            # only need when using user input
            queue_size = 10  # TODO: Maybe increase Queue size for subprcvecenv
            self.user_queue = Queue(maxsize=queue_size)
            self.response_queue = Queue()
            self.pref_interface, self.pref_process = start_pref_interface(self.user_queue, self.response_queue,
                                                                              queue_size,
                                                                              max_user_r=args.max_user_reward,
                                                                            synthetic_prefs=args.online_training=='perfect_user')

        elif self.online_training == 'real_user_files':
            self.waiting_files = 0
            self.max_waiting = 10
            self.total_saved_windows = 0
            self.total_loaded_windows = 0
            # dirs for saving / collecting windows
            self.save_dir = os.path.join(args.user_interface_dir, 'unlabeled')
            self.load_dir = os.path.join(args.user_interface_dir, 'labeled')
            self.resolved_dir = os.path.join(args.user_interface_dir, 'resolved')
            Path(self.save_dir).mkdir(exist_ok=True)
            Path(self.load_dir).mkdir(exist_ok=True)
            Path(self.resolved_dir).mkdir(exist_ok=True)
        
        self.max_user_windows = getattr(args, 'max_user_windows', 100000000)


    def reset_env(self, env_num):
        '''
        Reset the window buffer of a specific environment
        :param env_num: the index of the environment
        :return:
        '''
        self.window_buffers[env_num].reset_buffer()

    def get_inserted_windows(self):
        return self._inserted_windows

    def get_replay_size(self):
        return self.replay_buffer.get_replay_size()

    def get_replay_buffer(self):
        '''
        returns a pointer to the RP_replay buffer
        '''
        return self.replay_buffer

    def insert_step_data(self, user_data, rp_obs, vars, rp_rewards, dones, actions):
        '''
        All input arguments first dimension is N = num_envs

        :param user_data: current observation after being processed to user_image
        :param rp_obs: current observation after being processed to RP suitable obs
        :param vars: current variance of RP models
        :param rp_rewards: current RP predicted reward
        :param dones: current done flags of all envs
        :param actions: current actions for all envs
        :return:
        The function insert the data to the current window buffer, checks which window holds the
        trainable condition and add it to the RP replay buffer
        '''
        max_var = 0
        max_segment = None

        # Insert each env to its correlated Window Buffer
        for idx in range(self.num_envs):
            curr_user_data = user_data[idx]
            curr_rp_obs = rp_obs[idx]
            curr_var = vars[idx]
            curr_rp_reward = rp_rewards[idx]
            curr_actions = actions[idx]
            self.window_buffers[idx].insert(curr_user_data, curr_rp_obs, curr_var, curr_rp_reward, curr_actions)

        # Now check which window can be inserted to RP replay buffer
            done = dones[idx]
            window_segment, window_var = self.window_buffers[idx].check_if_current_window_trainable(done)
            if window_segment:
                if window_var > max_var:
                    max_var = window_var
                    max_segment = window_segment

        # Only take the window with the highest variance from all envs
        if max_segment:
            self.insert_windows(max_segment)

    def insert_windows(self, segment):

        # First, check if there is a log_dir for logging the trained sequence
        if self._log_dir:
          segement_file = os.path.join(self._log_dir, "trained_segments", f"segement_{self._inserted_windows}.pkl")
          with open(segement_file, "wb") as F:
               pickle.dump(segment, F, pickle.HIGHEST_PROTOCOL)

        self._inserted_windows += 1
        if self.online_training == 'real_user':
            if self._inserted_windows < self.max_user_windows:
                # send window to user evaluation
                self.user_queue.put(segment)
                # take response and insert to predictor buffer
                self._process_response_queue()

            # real user using files
        elif self.online_training == 'real_user_files':
            # TODO: need to implement
            self.save_user_file(segment)
            self.collect_user_files()
        else:
            if self.online_training == 'perfect_user':
                win_r = np.sum(segment.rewards)
            elif self.online_training == 'discrete_user':
                # mean reward fow window would be between -1 and 1
                # After taking mean, we want to discretisize the reward between
                # -1 and 1 to self.max_user_reward vals
                mean_r = np.mean(segment.rewards)
                mul = np.round(mean_r * (self.max_user_reward / 2.0))
                win_r = mul * (2.0 / self.max_user_reward)
            if self.noisy_user > 0:
                mu, sigma = 0, self.noisy_user
                noise = np.random.normal(mu, sigma, 1).item()
                win_r += noise

            # let RP train on window with real reward
            obs = np.stack(segment.obs)
            actions = np.stack(segment.actions)
            self.replay_buffer.insert(obs, win_r, actions)

    def _process_response_queue(self):
        while not self.response_queue.empty():
            obs, user_reward, actions, traj_mask = self.response_queue.get()  # block=False)
            # Window size is in win_size, no worries about padding
            self.replay_buffer.insert(obs, [user_reward], actions)

    # real user file interface
    def save_user_file(self, segment):
        print("save user file")
        seg_dict = {
            'frames': segment.frames,
            'obs': segment.obs,
            'actions': segment.actions,
            'rewards': segment.rewards
        }
        seg_file = os.path.join(self.save_dir, 'win_{}.pk'.format(self.total_saved_windows))
        lock = FileLock(seg_file + ".lock")
        with lock:
            with open(seg_file, 'wb') as handle:
                pickle.dump(seg_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.total_saved_windows += 1
        self.waiting_files += 1


    def collect_user_files(self):
        load_win_files = glob.glob(os.path.join(self.load_dir, 'win_*.pk'))
        while(len(load_win_files) > 0 or self.waiting_files > self.max_waiting):
            for file in load_win_files:
                print("load file: ", file)
                lock = FileLock(file + ".lock")
                with lock:
                    with open(file, 'rb') as handle:
                        seg_dict = pickle.load(handle)
                        self.replay_buffer.insert(seg_dict['obs'], [seg_dict['user_reward']], seg_dict['actions'])

                # move file to resolved
                os.rename(file, os.path.join(self.resolved_dir,
                                                  'resolved_win_{}.pk'.format(self.total_loaded_windows)))
                self.waiting_files -= 1

            load_win_files = glob.glob(os.path.join(self.load_dir, 'win_*.pk'))


