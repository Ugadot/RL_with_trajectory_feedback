import numpy as np
import os
import sys
import gym
from src.Reward_pred.reward_pred import Predictor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process, Queue
from src.utils.user_interface import PrefInterface, Segment, start_pref_interface
import wandb
from numpy_ringbuffer import RingBuffer

class RPWrapper(gym.Wrapper):
    """Wraps the environment to allow a modular transformation.
    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.
    .. note::
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """
    def __init__(self, env, args):
        super().__init__(env)

        # reward predictor
        self.device = args.device
        if args.online_training:
            load_path = None
        else:
            load_path = args.RP_path
        # TODO: check if can take from
        # Atari seems provide range of (-inf,inf)
        r_norm = args.env_max_r
        self.RP = Predictor(obs_space=self.observation_space,
                            threshold=4,  # TODO: add to arguments?
                            device=self.device,
                            model_load_path=load_path,
                            env_name="",
                            show_trajectory=None,
                            num_of_parallel_envs=1,
                            save_results_path=args.log_dir,
                            wandb_project=args.wandb,
                            r_norm_factor=r_norm)

        self.online_training = args.online_training
        self.user_reward = args.user_reward
        self._tot_steps = 0
        self.win_size = args.win_size
        self.rp_factor = args.rp_factor
        self.replace_reward = args.use_RP
        if self.online_training:
            self.replay_size = args.replay_size

            # TODO: Hagai, Change naming with refactor to RP_obs_replay_buffer etc.
            self.obs_buf = []
            self.r_buf = []
            self.masks_buf = []

            self.step_count = 0
            self._clear_current_buffer()  # restart current cyclic buffers
            self.train_interval = args.train_interval
            self.model_var_thresh = args.var_threshold
            self.window_ask_p = args.window_ask_p
            self._inserted_windows = 0
            self._total_windows = 0

            if args.user_reward:
                # only need when using user input
                queue_size = 10
                self.user_queue = Queue(maxsize=queue_size)
                self.response_queue = Queue()
                self.pref_interface, self.pref_process = start_pref_interface(self.user_queue, self.response_queue,
                                                                              queue_size,
                                                                              max_user_r=args.max_user_reward,
                                                                              synthetic_prefs=args.perfect_user)

            # self.pref_interface = PrefInterface(False, 10)
            # self.pref_process = Process(target=self.pref_interface.run, args=[self.user_queue, self.response_queue])
            # self.pref_process.start()

            # self.pref_interface.recv_segments(self.user_queue)
        self.wandb = args.wandb
        self.rp_batch_size = args.rp_batch_size

    def step(self, action):
        # call step on wrapped env
        obs, reward, done, info = self.env.step(action)
        if not 'wandb_log' in info.keys():
            info['wandb_log'] = {}

        # preprocess for reward predictor
        rp_obs = self.get_rp_ob(obs, reward, done, info)
        user_data = self.get_user_data(obs, reward, done, info)
        rp_reward = self.get_rp_reward(obs, reward, done, info)

        # predict - add window size 1 dimension
        r_int, var = self.RP.predict(np.expand_dims(rp_obs, axis=0))

        if type(reward) == float:
            # convert predict res from array (size 1) to float
            r_int = float(r_int)

        if self.online_training:
            # insert to training buffers
            self.user_data_buffer.append(user_data)
            self.window_states_buffer.append(rp_obs)
            self.predicted_var_buffer.append(var)
            self.real_reward_buffer.append(rp_reward)

            # check if need to train on current window
            self._total_windows += 1
            nd_vars_buffer = np.array(self.predicted_var_buffer)
            avg_win_var = np.mean(nd_vars_buffer)

            # self.window_ask_p is a probability of taking the window into account without the VAR being larger than treshold
            large_var = avg_win_var > self.model_var_thresh or np.random.random() <= self.window_ask_p
            info['wandb_log']["RP_Average_window_var"] = avg_win_var
            # TODO: Remove total_windows in general??
            # info['wandb_log']["Total_windows"] = self._total_windows
            # Temp removal of done for checks
            #add_to_train_data_condition = (len(self.user_data_buffer) == self.win_size or done) and large_var
            add_to_train_data_condition = (len(self.user_data_buffer) == self.win_size) and large_var

            # train on current window
            if add_to_train_data_condition:
                # Push current Window to User Queue
                current_segment = Segment(np.array(self.user_data_buffer),
                                          np.array(self.window_states_buffer),
                                          np.array(self.real_reward_buffer))

                self._inserted_windows += 1
                if ('global_wandb_log' not in info.keys()):
                    info['global_wandb_log'] = {}
                info['global_wandb_log']["RP_Train/Train_windows"] = self._inserted_windows
                if self.user_reward:
                    # send window to user evaluation

                    self.user_queue.put(current_segment)
                    # take response and insert to predictor buffer
                    self._process_response_queue()
                else:
                    # let RP train on window with real reward
                    win_r = sum(self.real_reward_buffer)
                    self._add_current_window_to_replay_buffer(win_r)

                # clear buffer after processing window
                self._clear_current_buffer()

            # RP training is orthogonal to add_to_train_data_condition
            self.step_count += 1
            # Check Response_queue to see if user answered 16 windows already
            if self.step_count >= self.train_interval and len(self.obs_buf) > self.rp_batch_size: #TODO: What is the right number?
                self.RP.one_batch_train(np.stack(self.obs_buf), np.array(self.r_buf, dtype=float), self.device, self.rp_batch_size)
                self.step_count = 0


        rp_total_reward = reward + self.rp_factor * r_int
        info['rp_predicted_reward'] = r_int
        info['rp_total_reward'] = rp_total_reward
        info['wandb_log']['rp_predicted_reward'] = r_int
        info['wandb_log']['rp_total_reward'] = rp_total_reward
        reward = rp_total_reward if self.replace_reward else reward
        return obs, reward, done, info

    def get_rp_ob(self, ob, reward, done, info):
        '''obeservation preprocessing for reward predictor'''
        rp_ob = ob

        return rp_ob

    def get_user_data(self, ob, reward, done, info):
        '''implement if need to add aditional info to predictor'''
        img = 0
        if self.spec.id == 'random_rooms-v1':
            img = self.unwrapped.get_current_img()
        return img

    def get_rp_reward(self, ob, reward, done, info):
        '''reward preprocessing for reward predictor
            if recieved reward in info use it
            otherwise use original reward
        '''
        assert 'rp_learned_reward' in info.keys(), "RP did not get reward to learn in info"
        R = info['rp_learned_reward']
        return R

    # This function is called only if user_reward is false
    def _add_current_window_to_replay_buffer(self, window_reward):

        # If we got here using done and the window size is not with win_size, we need to pad the window
        if len(self.user_data_buffer) < self.win_size:
            obs = np.stack(self.window_states_buffer)
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
            self.obs_buf.append(obs)
            self.r_buf.append([window_reward])
            self.masks_buf.append(traj_mask)

        else:  # Window size is in win_size, no worries about padding
            self.obs_buf.append(np.stack(self.window_states_buffer))
            self.r_buf.append([window_reward])
            self.masks_buf.append((np.ones_like(self.obs_buf[-1]))) # Should be with the same shape

    def _clear_current_buffer(self):
        self.user_data_buffer = RingBuffer(capacity=self.win_size, dtype=np.ndarray)
        self.window_states_buffer = RingBuffer(capacity=self.win_size, dtype=np.ndarray)
        self.predicted_var_buffer = RingBuffer(capacity=self.win_size, dtype=np.float)
        self.real_reward_buffer = RingBuffer(capacity=self.win_size, dtype=np.float)

    # Take user reward from response queue and insert to replay buffer
    def _process_response_queue(self):
        while (not self.response_queue.empty()):
            obs, user_reward, traj_mask = self.response_queue.get()  # block=False)

            # If we got here using done and the window size is not with win_size, we need to pad the window
            if len(obs) < self.win_size:
                traj_mask_ones = traj_mask
                dummy_obs_shape = list(obs.shape)
                dummy_obs_shape[0] = self.win_size - dummy_obs_shape[0] # Change length of window to the remaining needed
                dummy_obs_shape = tuple(dummy_obs_shape)
                dummy_obs = np.zeros(dummy_obs_shape)
                traj_mask_zeros = np.zeros_like(dummy_obs)
                obs = np.concatenate((obs, dummy_obs))
                traj_mask = np.concatenate((traj_mask_ones, traj_mask_zeros))
                assert (obs.shape == traj_mask.shape)

                # Insert padded data to replay buffer
                self.obs_buf.append(obs)
                self.r_buf.append([user_reward])
                self.masks_buf.append(traj_mask)

            else:  # Window size is in win_size, no worries about padding
                self.obs_buf.append(obs) # TODO: Maybe np.stack is needed? currently not
                self.r_buf.append([user_reward])
                self.masks_buf.append(traj_mask)

            # extract old experience from replay buffer if got full
            if len(self.obs_buf) > self.replay_size:
                self.obs_buf = self.obs_buf[1:]
                self.r_buf = self.r_buf[1:]
                self.masks_buf = self.masks_buf[1:]


# TODO: Check if used
def _plot_window(win_size, images):
    fig = plt.figure()

    ims = []
    for i in range(win_size):
        if (i + 1 < len(images)):
            im = plt.imshow(np.flip(images[i], axis=0), animated=True)
            ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.show()