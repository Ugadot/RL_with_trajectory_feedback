import numpy as np
import torch

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

import os
import sys
from multiprocessing import Process, Queue
from src.utils.user_interface import PrefInterface, Segment

file_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(file_dir)
from src.utils.user_interface import PrefInterface, Segment, start_pref_interface
from src.Reward_pred.reward_pred import Predictor
from numpy_ringbuffer import RingBuffer
from src.window_buffer.WindowBuffer import WindowBuffer


class RPSubprocVecEnv(SubprocVecEnv):
    """
      SubprocVecEnv that uses RP (instead of RP Wrapper?)
    """
    def __init__(self, env_fns, start_method=None, args=None):
        super(RPSubprocVecEnv, self).__init__(env_fns, start_method)
        self.device = args.device
        if args.online_training != "":
            load_path = None
        else:
            load_path = args.RP_path
        # TODO: check if can take from
        # Atari seems provide range of (-inf,inf)
        r_norm = args.env_max_r

        action_space = None
        self._predict_with_action = args.predict_with_action
        if self._predict_with_action:
            action_space = self.action_space

        self.RP = Predictor(obs_space=self.observation_space,
                            device=self.device,
                            model_load_path=load_path,
                            show_trajectory=None,  # TODO: Is this needed?
                            save_results_path=args.log_dir,
                            wandb_project=args.wandb,
                            r_norm_factor=r_norm,
                            action_prediction_space=action_space)

        self.online_training = args.online_training
        self.max_user_reward = args.max_user_reward
        self._tot_steps = 0
        self.win_size = args.win_size
        self.rp_factor = args.rp_factor
        self.replace_reward = args.use_RP
        if self.online_training != "":
            # Replay buffers are data saved for RP training, if there are several environments the windows are split
            # and inserted differently to the replay buffer
            self.window_buffer = WindowBuffer(args=args)

            self.step_count = 0
            self.train_interval = args.train_interval

        self.wandb = args.wandb
        self.rp_batch_size = args.rp_batch_size
        # save prev ob for RP
        self.prev_ob = None
        self.args = args

   
    def step(self, actions: np.ndarray):
        # call step on wrapped env
        obs, r_ext, dones, infos = super().step(actions)

        for info in infos:
            if not 'wandb_log' in info.keys():
                info['wandb_log'] = {}


        # preprocess for reward predictor
        rp_obs = self.get_rp_ob(obs, r_ext, dones, infos)
        user_datas = self.get_user_data(obs, r_ext, dones, infos)
        rp_rewards = self.get_rp_reward(obs, r_ext, dones, infos)

        # [Uri] predict - Maybe need to add window size 1 dimension (but because its vecEnv i dont think so)
        if self._predict_with_action:
            r_ints, vars = self.RP.predict(rp_obs, actions)
        else:
            r_ints, vars = self.RP.predict(rp_obs)

        if self.online_training != "":
            # insert to training buffers
            self.window_buffer.insert_step_data(user_datas, rp_obs, vars, rp_rewards, dones, actions)

            # Currently updating genral information about RP to env zero only!
            inserted_windows = self.window_buffer.get_inserted_windows()
            if ('global_wandb_log' not in infos[0].keys()):
                infos[0]['global_wandb_log'] = {}
            infos[0]['global_wandb_log']["RP_Train/Train_windows"] = inserted_windows

            # Train RP if possible and needed
            self.step_count += 1
            rp_replay_buffer_size = self.window_buffer.get_replay_size()
            if self.step_count >= self.train_interval and rp_replay_buffer_size > self.rp_batch_size:  # TODO: What is the right number?
                replay_buffer_for_train = self.window_buffer.get_replay_buffer()
                #self.RP.one_batch_train(np.stack(self.RP_obs_replay_buffer), np.array(self.RP_rewards_replay_buffer, dtype=float), self.device, self.rp_batch_size)
                self.RP.one_batch_train_temp_for_subproc(replay_buffer_for_train, self.device, self.rp_batch_size)
                self.step_count = 0

        rp_total_reward = r_ext + self.rp_factor * r_ints
        for index, info in enumerate(infos):
            info['wandb_log']["rp_predicted_reward"] = r_ints[index]
            # info['wandb_log']["rp_total_reward"] = rp_total_reward[index]
            info['wandb_log']["rp_var"] = vars[index]
            info['r_int'] = r_ints[index]

        # TODO: remove
        for index, info in enumerate(infos):
            diff = info['wandb_log']["rp_predicted_reward"] - info['rp_learned_reward']
            info['wandb_log']['rp_diff'] = diff

        if self.args.rp_norm:
            rewards = r_ext
        else:
            rewards = rp_total_reward if self.replace_reward else r_ext

        return obs, rewards, dones, infos


    # This function is called only if user_reward is false
    def _add_current_window_to_replay_buffer(self, window_reward, index):
        # If we got here using done and the window size is not with win_size, we need to pad the window
        obs = np.stack(self.window_states_buffer).take(indices=index, axis=1)
        env_window_len = obs.shape[0]
        if env_window_len < self.win_size:
            traj_mask_ones = np.ones_like(obs)
            dummy_obs_shape = list(obs.shape)
            dummy_obs_shape[0] = self.win_size - dummy_obs_shape[0]  # Change length of window to the remaining needed
            dummy_obs_shape = tuple(dummy_obs_shape)
            dummy_obs = np.zeros(dummy_obs_shape)
            traj_mask_zeros = np.zeros_like(dummy_obs)
            obs = np.concatenate((obs, dummy_obs))
            traj_mask = np.concatenate((traj_mask_ones, traj_mask_zeros))
            assert (obs.shape == traj_mask.shape)

            # Insert padded data to replay buffer
            self.RP_obs_replay_buffer.append(obs)
            self.RP_rewards_replay_buffer.append([window_reward])
            self.RP_masks_replay_buffer.append(traj_mask)

        else:  # Window size is in win_size, no worries about padding
            self.RP_obs_replay_buffer.append(obs)
            self.RP_rewards_replay_buffer.append([window_reward])
            self.RP_masks_replay_buffer.append((np.ones_like(self.RP_obs_replay_buffer[-1])))  # Should be with the same shape


    # def get_rp_ob(self, obs, rewards, dones, infos):
    #     '''obeservation preprocessing for reward predictor'''
    #     rp_obs = obs
    #     return rp_obs

 
    def get_rp_ob(self, obs, rewards, dones, infos):
        '''obeservation preprocessing for reward predictor
            Return previous observation
            Since this class wrapps SubProcVecEnv, when there is done we get
             the observation after reset:
             https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py'''
        rp_obs = self.prev_ob if self.prev_ob is not None else obs
        self.prev_ob = obs
        return rp_obs
        return rp_obs


    def get_user_data(self, obs, rewards, dones, infos):
        '''implement if need to add aditional info to predictor'''
        imgs = self.env_method("get_current_img")
        # TODO: check if this is working
        return imgs

    def get_rp_reward(self, obs, rewards, dones, infos):
        '''reward preprocessing for reward predictor
            if recieved reward in info use it
            otherwise use original reward
        '''
        rp_rewards = [info['rp_learned_reward'] for info in infos]
        return rp_rewards

    def _clear_current_buffer(self):
        '''
        Each Ring buffer entry contains ndarray of N (number of envs) X data_size
        RingBuffer dims are: (window_size X envs X (data_size))
        Where data_size can be a single scalar or np.matrix
        '''
        self.user_data_buffer = RingBuffer(capacity=self.win_size, dtype=np.ndarray)
        self.window_states_buffer = RingBuffer(capacity=self.win_size, dtype=np.ndarray)
        self.predicted_var_buffer = RingBuffer(capacity=self.win_size, dtype=np.ndarray)
        self.real_reward_buffer = RingBuffer(capacity=self.win_size, dtype=np.ndarray)
        self.ring_buffer_lens = np.zeros(self.num_envs, dtype=int)
        # self.MAX_WINDOWS is the maximum length of the win_size per enviroment
        self.MAX_WINDOWS = self.win_size * np.ones_like(self.ring_buffer_lens)

    def _clear_current_buffer_by_index(self, index):
        '''
        Clear the data of a specific env data (update len of window and zero all variance info)
        '''
        for step in range(self.win_size):
            self.predicted_var_buffer[step][index] = 0
        self.ring_buffer_lens[index] = 0

    def _process_response_queue(self):
        while (not self.response_queue.empty()):
            obs, user_reward, traj_mask = self.response_queue.get()  # block=False)

            # If we got here using done and the window size is not with win_size, we need to pad the window
            if len(obs) < self.win_size:
                traj_mask_ones = traj_mask
                dummy_obs_shape = list(obs.shape)
                dummy_obs_shape[0] = self.win_size - dummy_obs_shape[
                    0]  # Change length of window to the remaining needed
                dummy_obs_shape = tuple(dummy_obs_shape)
                dummy_obs = np.zeros(dummy_obs_shape)
                traj_mask_zeros = np.zeros_like(dummy_obs)
                obs = np.concatenate((obs, dummy_obs))
                traj_mask = np.concatenate((traj_mask_ones, traj_mask_zeros))
                assert (obs.shape == traj_mask.shape)

                # Insert padded data to replay buffer
                self.RP_obs_replay_buffer.append(obs)
                self.RP_rewards_replay_buffer.append([user_reward])
                self.RP_masks_replay_buffer.append(traj_mask)

            else:  # Window size is in win_size, no worries about padding
                self.RP_obs_replay_buffer.append(obs)  # TODO: Maybe np.stack is needed? currently not
                self.RP_rewards_replay_buffer.append([user_reward])
                self.RP_masks_replay_buffer.append(traj_mask)

            # extract old experience from replay buffer if got full
            if len(self.RP_obs_replay_buffer) > self.replay_size:
                self.RP_obs_replay_buffer = self.RP_obs_replay_buffer[1:]
                self.RP_rewards_replay_buffer = self.RP_rewards_replay_buffer[1:]
                self.RP_masks_replay_buffer = self.RP_masks_replay_buffer[1:]

