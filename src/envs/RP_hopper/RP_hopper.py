import numpy as np
from gym.envs.mujoco.hopper import HopperEnv
from src.Reward_pred.reward_pred import Predictor

from numpy_ringbuffer import RingBuffer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter

class RPHopperEnv(HopperEnv):
    def __init__(self, Hopper_args):
        super(RPHopperEnv, self).__init__()
        
        self.tot_reward = 0
        self.tot_sparse_reward = 0
        self.nsteps = 0
        self.sparse_reward_value = Hopper_args.sparse_reward_value
        self.use_sparse_reward = Hopper_args.use_sparse_reward

        # distance reward vars
        self.reward_distance = Hopper_args.reward_distance
        self.first_pos = None
        self.abs_first_pos = None
        
        # accu reward vars
        self.accumulative_threshold = Hopper_args.accumulative_threshold
        self.temp_accu_reward = 0 
        
        # RP args (for online training)
        self.RP = None
        self.predict_with_action = Hopper_args.predict_with_action

        if Hopper_args.reward_pred or Hopper_args.online_training:
            self.state_index = 0
            if Hopper_args.online_training:
                load_path = None
            else:
                load_path = Hopper_args.reward_pred

            obs_size = self.observation_space.sample().size
            action_size = self.action_space.sample().size
            fc_layer_size = (obs_size + action_size) if self.predict_with_action else obs_size
                        
            self.RP = Predictor(threshold = 4,  # TODO: add to arguments?
                                device = Hopper_args.device,
                                model_load_path = load_path,
                                env_name = "",
                                show_trajectory = None,
                                num_of_parallel_envs = 1,
                                siamese_type = " linear",
                                input_fc_layer_size = fc_layer_size,
                                save_results_path = Hopper_args.log_dir)
            
            self.rp_factor = Hopper_args.rp_factor

        self.online_training = Hopper_args.online_training
        self.device = Hopper_args.device
        self._tot_steps = 0
        if self.online_training:
            self.replay_size = Hopper_args.replay_size

            self.obs_buf = []
            self.r_buf = []

            self.step_count = 0
            self.win_size = Hopper_args.win_size
            self._clear_current_buffer() #restart current cyclic buffers
            self.train_interval = Hopper_args.train_interval
            self.model_var_thresh = Hopper_args.var_threshold
            self._inserted_windows = 0
            self._asked_windows = 0
        if self.RP:
            self.tensorboard_writer = self.RP.get_tensorBoard_Writer()
        else:
            self.tensorboard_writer = None
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
        self._asked_windows += 1
        nd_vars_buffer = np.array(self.predicted_var_buffer)
        if (self.tensorboard_writer and self._tot_steps % 50 == 0):  # TODO: Maybe 50 should be var?
            self.tensorboard_writer.add_scalar("Average_window_var", np.mean(nd_vars_buffer), self._tot_steps)
            self.tensorboard_writer.add_scalar("Asked_windows",  self._asked_windows, self._tot_steps)
        return np.mean(nd_vars_buffer) > threshold
        

    def _show_window_images(self):
       
        nd_images_buffer = np.array(self.images_buffer)
        nd_rewards_buffer = np.array(self.real_reward_buffer)
        
        #p = Process(target=_plot_window, args=[self.win_size, nd_images_buffer])
        #p.start()
        
        #try:
        #    reward = float(input("Enter the window wanted value (real value is {}): ".format(np.sum(nd_rewards_buffer))))
        #except:
        #    print("didn't get a number, return reward=0")
        #    reward = 0

        #p.terminate()
        reward = np.sum(nd_rewards_buffer)
        self._inserted_windows += 1
        nd_vars_buffer = np.array(self.predicted_var_buffer)
        print (str(self._inserted_windows) + "\t" + str(np.mean(nd_vars_buffer)))
        return reward  

    def _get_replay_buffer_size(self):
        return len(self.obs_buf)
    
    def _add_current_window_to_replay_buffer(self, window_reward):
        self.obs_buf.append(np.stack(self.window_states_buffer))
        self.r_buf.append([window_reward])

        if len(self.obs_buf) > self.replay_size:
            self.obs_buf = self.obs_buf[1:]
            self.r_buf = self.r_buf[1:]
     
    def step(self, a):
        #TODO: this step is only for uniform decision distance experiment
        #uniform_step = np.random.uniform(-1,1,3)
        #ob, reward, done, _ = super(RPHopperEnv, self).step(uniform_step)

        prev_ob = self._get_obs()

        ob, reward, done, _ = super(RPHopperEnv, self).step(a)
        #a return for the first step when env is initialized
        if (not hasattr(self, 'nsteps')):
            return ob, reward, done, _

        self.nsteps += 1
        self._tot_steps += 1
        self.tot_reward  += reward
        self.temp_accu_reward  += reward
        dense_reward = reward

        # Calc accu_reward
        accu_reward = 0
        if (self.temp_accu_reward >= self.accumulative_threshold):
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

       

        self.tot_sparse_reward += sparse_reward       
        
        info = {}
        info['special_r'] = dense_reward

        if done or self.nsteps>999:
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
        
        if self.online_training:
            
            data = self.sim.render(420,380) #TODO: we removed a if that checks if data is None (maybe this can cause problems)
            self._add_data_to_current_buff(data, ob_2_RP, var, dense_reward)
            
            if (self._get_current_buffer_len() == self.win_size and self._is_average_var_large(self.model_var_thresh)):
                reward = self._show_window_images()
                self._add_current_window_to_replay_buffer(reward)
                self._clear_current_buffer()

            self.step_count += 1
            if self.step_count >= self.train_interval and self._get_replay_buffer_size() > 16: #TODO: What is the right number?
                self._train_rp()
                self.step_count = 0
            
            if done:
                self._clear_current_buffer()
        
            if (self.tensorboard_writer and self._tot_steps % 50 == 0):  # TODO: Maybe 50 should be var?
                self.tensorboard_writer.add_scalar("User_windows", self._inserted_windows, self._tot_steps)
            
        return ob, R, done, info
  
    def _train_rp(self):
        self.RP.one_batch_train(np.stack(self.obs_buf), np.array(self.r_buf, dtype=float), self.device, 4)

    def _rp_var_reward(self, var):
        # return reward depending on rp models disagreement
        return 0
     
    def reset_model(self):
        self.temp_accu_reward = 0
        self.tot_reward = 0
        self.tot_sparse_reward = 0
        self.first_pos = None
        self.nsteps = 0
        return super(RPHopperEnv, self).reset_model()



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