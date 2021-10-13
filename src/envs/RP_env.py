import numpy as np
import os
import sys

from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco import  HopperEnv, AntEnv, InvertedDoublePendulumEnv,  HalfCheetahEnv

from src.Reward_pred.reward_pred import Predictor

from numpy_ringbuffer import RingBuffer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process, Queue
from torch.utils.tensorboard import SummaryWriter

from src.utils.user_interface import PrefInterface, Segment
import wandb

def get_rp_env(args):
    if args.env_name == 'pendulum':
        env = InvertedPendulumEnv
    elif args.env_name == 'double_pendulum':
        env = InvertedDoublePendulumEnv
    elif args.env_name == 'hopper':
        env = HopperEnv
    if args.env_name == 'ant':
        env = AntEnv
    if args.env_name == 'half_cheetah':
        env = HalfCheetahEnv
        
    
    class RPEnv(env):
        def __init__(self, args):
            super(RPEnv, self).__init__()
            
            self.tot_reward = 0
            self.tot_sparse_reward = 0
            self.nsteps = 0
            self.sparse_reward_value = args.sparse_reward_value
            self.use_sparse_reward = args.use_sparse_reward

            # distance reward vars
            self.reward_distance = args.reward_distance
            self.first_pos = None
            self.abs_first_pos = None
            
            # accu reward vars
            self.accumulative_threshold = args.accumulative_threshold
            self.temp_accu_reward = 0 
            
            # RP args (for online training)
            self.RP = None
            self.predict_with_action = args.predict_with_action

            if args.reward_pred or args.online_training:
                self.state_index = 0
                if args.online_training:
                    load_path = None
                else:
                    load_path = args.reward_pred

                obs_size = self.observation_space.sample().size
                action_size = self.action_space.sample().size
                fc_layer_size = (obs_size + action_size) if self.predict_with_action else obs_size

                r_norm = args.env_max_r

                #  TODO: Change arguments to work wiht observation space input to RP
                self.RP = Predictor(device=args.device,
                                    model_load_path=load_path,
                                    show_trajectory = None,
                                    siamese_type = " linear",  # TODO: Delete this
                                    input_fc_layer_size = fc_layer_size,  # TODO: Delete this
                                    save_results_path = args.log_dir,
                                    wandb_project=args.wandb,
                                    r_norm_factor=r_norm)
                
                self.rp_factor = args.rp_factor

            self.online_training = args.online_training
            self.user_reward = args.user_reward
            self.device = args.device
            self._tot_steps = 0
            self.win_size = args.win_size
            if self.online_training:
                self.replay_size = args.replay_size

                self.obs_buf = []
                self.r_buf = []
                self.masks_buf = []

                self.step_count = 0
                self._clear_current_buffer() #restart current cyclic buffers
                self.train_interval = args.train_interval
                self.model_var_thresh = args.var_threshold
                self._inserted_windows = 0
                self._total_windows = 0
                
                if args.user_reward:
                    # onlt neede when using user input
                    queue_size=10
                    self.user_queue = Queue(maxsize=queue_size)
                    self.response_queue = Queue()
                    self.pref_interface, self.pref_process = start_pref_interface(self.user_queue, self.response_queue,queue_size, 
                                                                                max_user_r=args.max_user_reward,
                                                                                synthetic_prefs=args.perfect_user)
                
                #self.pref_interface = PrefInterface(False, 10)
                #self.pref_process = Process(target=self.pref_interface.run, args=[self.user_queue, self.response_queue])
                #self.pref_process.start()
                
                #self.pref_interface.recv_segments(self.user_queue)

            if self.RP:
                self.tensorboard_writer = self.RP.get_tensorBoard_Writer()
            else:
                self.tensorboard_writer = None
            
            self.wandb = args.wandb
            self.rp_batch_size = args.rp_batch_size
        
            
        # End of RP args
      
        def _get_current_buffer_len(self):
            return len(self.images_buffer)

        def _add_data_to_current_buff(self, image, obs, var, real_r):
            self.images_buffer.append(image)
            self.window_states_buffer.append(obs)
            self.predicted_var_buffer.append(var)
            self.real_reward_buffer.append(real_r)
        
        def _clear_current_buffer(self):
            self.images_buffer = RingBuffer(capacity=self.win_size, dtype=np.ndarray)
            self.window_states_buffer = RingBuffer(capacity=self.win_size, dtype=np.ndarray)
            self.predicted_var_buffer = RingBuffer(capacity=self.win_size, dtype=np.float)
            self.real_reward_buffer = RingBuffer(capacity=self.win_size, dtype=np.float)
        
        def _is_average_var_large(self, threshold):
            self._total_windows += 1
            nd_vars_buffer = np.array(self.predicted_var_buffer)
            avg_win_var = np.mean(nd_vars_buffer)
            if (self.tensorboard_writer and self._tot_steps % 50 == 0):  # TODO: Maybe 50 should be var?
                self.tensorboard_writer.add_scalar("RP/Average_window_var", avg_win_var, self._tot_steps)
                self.tensorboard_writer.add_scalar("RP/Total_windows",  self._total_windows, self._tot_steps)
            # if self.wandb:
            #     wandb.log({"RP/Average_window_var": }) #, step=self._tot_steps)
            #     wandb.log({"RP/Total_windows":  self._total_windows}) #, step=self._tot_steps)
            return np.mean(nd_vars_buffer) > threshold, avg_win_var
            

        # def _show_window_images(self):
        
        #     nd_images_buffer = np.array(self.images_buffer)
        #     nd_rewards_buffer = np.array(self.real_reward_buffer)
            
        #     p = Process(target=_plot_window, args=[self.win_size, nd_images_buffer])
        #     p.start()
            
        #     try:
        #         reward = float(input("Enter the window wanted value (real value is {}): ".format(np.sum(nd_rewards_buffer))))
        #     except:
        #         print("didn't get a number, return reward=0")
        #         reward = 0

        #     p.terminate()
        #     #reward = np.sum(nd_rewards_buffer)
        #     self._inserted_windows += 1
        #     nd_vars_buffer = np.array(self.predicted_var_buffer)
        #     print (str(self._inserted_windows) + "\t" + str(np.mean(nd_vars_buffer)))
        #     return reward  

        def _get_replay_buffer_size(self):
            return len(self.obs_buf)
        
        # This function is called only if user_reward is false
        def _add_current_window_to_replay_buffer(self, window_reward):
            self.obs_buf.append(np.stack(self.window_states_buffer))
            self.r_buf.append([window_reward])
            self.masks_buf.append((np.ones_like(self.obs_buf[-1]))) # Should be with the same shape

            if len(self.obs_buf) > self.replay_size:
                self.obs_buf = self.obs_buf[1:]
                self.r_buf = self.r_buf[1:]
                self.masks_buf = self.masks_buf[1:]
        
        def step(self, a):
            #TODO: this step is only for uniform decision distance experiment
            #uniform_step = np.random.uniform(-1,1,3)
            #ob, reward, done, _ = super(RPEnv, self).step(uniform_step)
            prev_ob = self._get_obs()

            ob, reward, done, info = super(RPEnv, self).step(a)
            #a return for the first step when env is initialized
            if (not hasattr(self, 'nsteps')):
                return ob, reward, done, info
            
            info['wandb_log'] = {}
            # info['wandb_log']['env_reward'] = reward

            self.nsteps += 1
            self._tot_steps += 1
            # if self.wandb:
            #     wandb.log({"total_steps": self._tot_steps}) #, step=self._tot_steps)
            self.tot_reward  += reward
            self.temp_accu_reward  += reward
            dense_reward = reward
            info['wandb_log']['env_episode_reward'] = self.tot_reward
            info['wandb_log']['episode_steps'] = self.nsteps
            info['wandb_log']['avg_env_reward'] = self.tot_reward / self.nsteps

            # Calc accu_reward
            accu_reward = 0
            if (self.use_sparse_reward and self.use_sparse_reward == 'accumulative' and self.temp_accu_reward >= self.accumulative_threshold):
                # zero acc when reach threshold
                accu_reward = self.sparse_reward_value
                self.temp_accu_reward = 0
            
            # Calc distance_reward
            distance_reward = 0
            if (self.reward_distance != None and self.sparse_reward_value != None):
                if (self.first_pos == None):
                    self.first_pos = self.sim.data.qpos[0]
                    self.abs_first_pos = self.sim.data.qpos[0]
                elif (self.sim.data.qpos[0] - self.first_pos >= self.reward_distance):
                    #print ("Reached X")
                    distance_reward = self.sparse_reward_value
                    self.first_pos = self.sim.data.qpos[0]

            sparse_reward = 0
            if (self.use_sparse_reward and self.use_sparse_reward == 'accumulative'):
                sparse_reward = accu_reward
                reward = sparse_reward
            elif (self.use_sparse_reward and self.use_sparse_reward == 'distance'):
                sparse_reward = distance_reward
                reward = sparse_reward
            elif (self.use_sparse_reward and self.use_sparse_reward == 'window' and self.nsteps % self.win_size == 0):
                # return window average reward every WIN_SIZE steps
                # zero acc
                sparse_reward = self.temp_accu_reward / self.win_size
                reward = sparse_reward
                self.temp_accu_reward = 0

            self.tot_sparse_reward += sparse_reward       
            
            info['wandb_log']['sparse_episode_reward'] = self.tot_sparse_reward
            info['wandb_log']['avg_sparse_reward'] = self.tot_sparse_reward / self.nsteps
            info['wandb_log']['distance'] = self.sim.data.qpos[0] - self.abs_first_pos
            # info = {}
            # info['special_r'] = dense_reward

            if done or self.nsteps>999:
                # done = True
                info['episode'] = {'r': self.tot_reward, 'l': self.nsteps, 'sparse_r': self.tot_sparse_reward,
                'distance': self.sim.data.qpos[0] - self.abs_first_pos}
            # return ob, reward, done, info
            
            if not self.RP:
                return  ob, reward, done, info
            
            # RP Part
            if self.predict_with_action:
                ob_2_RP = np.concatenate((ob, a), axis=0)
            else:
                ob_2_RP = ob
            
            r_int, var = self.RP.predict(ob_2_RP)
            r_unk = self._rp_var_reward(var)
            R = reward + self.rp_factor * (r_int + r_unk)
            
            self.tot_r_int += r_int
            self.tot_r_unk += r_unk
            
            info['wandb_log']['RP/episode_r_int'] = self.tot_r_int
            info['wandb_log']['RP/episode_r_unk'] = self.tot_r_unk
            info['wandb_log']['RP/avg_r_int'] = self.tot_r_int / self.nsteps
            info['wandb_log']['RP/avg_r_unk'] = self.tot_r_unk / self.nsteps

            if self.online_training:
                data = self.sim.render(420,380) #TODO: we removed a if that checks if data is None (maybe this can cause problems)
                self._add_data_to_current_buff(data, ob_2_RP, var, dense_reward)
                large_var,  avg_win_var = self._is_average_var_large(self.model_var_thresh)
                info['wandb_log']["RP/Average_window_var"] = avg_win_var
                info['wandb_log']["RP/Total_windows"] =   self._total_windows
                
                if (self._get_current_buffer_len() == self.win_size and large_var):
                # Push current Window to User Queue
                    current_segment = self._create_segment()
                    self._inserted_windows += 1
                    if self.user_reward:
                        # send window to user evaluation
                        self.user_queue.put(current_segment)
                    
                        #temp usage
                        self._process_response_queue()
                        #reward = self._show_window_images()
                        #self._add_current_window_to_replay_buffer(reward)

                        #Check for user answers in response_queue
                        #if (self.response_queue.get(block=False)): # TODO: maybe implement with True to avoid training with bad RP ??                
                        #self._process_response_queue()
                    else:
                        # let TP train on window with real reward

                        
                        win_r = sum(self.real_reward_buffer)
                        # win_r = sum(self.real_reward_buffer) / len(self.real_reward_buffer)
                        self._add_current_window_to_replay_buffer(win_r)
                    
                    # clear buffer after processing window
                    self._clear_current_buffer()


                self.step_count += 1
                # Check Response_queue to see if user answered 16 windows already
                if self.step_count >= self.train_interval and self._get_replay_buffer_size() > self.rp_batch_size: #TODO: What is the right number?
                    self._train_rp()
                    self.step_count = 0
                
                if done:
                    self._clear_current_buffer()
            
                if (self.tensorboard_writer and self._tot_steps % 50 == 0):  # TODO: Maybe 50 should be var?
                    self.tensorboard_writer.add_scalar("User_windows", self._inserted_windows, self._tot_steps)
                # if self.wandb and self._tot_steps % 50 == 0:
                #     wandb.log({"RP/User_windows": self._inserted_windows}) #, step=self._tot_steps)
                info['wandb_log']['User_windows'] = self._inserted_windows

            return ob, R, done, info
    
        def _train_rp(self):
            self.RP.one_batch_train(np.stack(self.obs_buf), np.array(self.r_buf, dtype=float), np.stack(self.masks_buf) ,self.device, self.rp_batch_size)

        def _rp_var_reward(self, var):
            # return reward depending on rp models disagreement
            return 0

        # Change init to get list already
        def _create_segment(self):
            new_segment = Segment(np.array(self.images_buffer), np.array(self.window_states_buffer), np.array(self.real_reward_buffer))
            return new_segment
        
        # Take user reward from response queue and insert to replay buffer
        def _process_response_queue(self):
            while (not self.response_queue.empty()):
                obs, user_reward, traj_mask = self.response_queue.get()  # block=False)
                #self.obs_buf.append(np.stack(obs)) # TODO: Maybe dont do np.stack
                self.obs_buf.append(obs) # TODO: Maybe dont do np.stack
                self.r_buf.append([user_reward])
                self.masks_buf.append(traj_mask)
                if len(self.obs_buf) > self.replay_size:
                    self.obs_buf = self.obs_buf[1:]
                    self.r_buf = self.r_buf[1:]
                    self.masks_buf = self.masks_buf[1:]    

        def reset_model(self):
            self.temp_accu_reward = 0
            self.tot_reward = 0
            self.tot_r_int = 0
            self.tot_r_unk = 0
            self.tot_sparse_reward = 0
            self.first_pos = None
            self.nsteps = 0
            return super(RPEnv, self).reset_model()

    return RPEnv(args)


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

def _plot_window(win_size, images):
    fig = plt.figure()

    ims = []
    for i in range(win_size):
        if ( i + 1 < len(images)):
            im = plt.imshow(np.flip(images[i], axis=0), animated=True)
            ims.append([im])
  
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.show()