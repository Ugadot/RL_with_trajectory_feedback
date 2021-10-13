import numpy as np
from gym.envs.mujoco.ant import AntEnv
from src.Reward_pred.reward_pred import Predictor

class RPAntEnv(AntEnv):
    def __init__(self, Ant_args):  
        
        super(RPAntEnv, self).__init__()

        self.tot_reward = 0
        self.tot_sparse_reward = 0
        self.nsteps = 0
        self.sparse_reward_value = Ant_args.sparse_reward_value
        self.use_sparse_reward = Ant_args.use_sparse_reward
        
        # distance reward vars
        self.reward_distance = Ant_args.reward_distance
        self.first_pos = None
        self.abs_first_pos = None
        
        # accu reward vars
        self.accumulative_threshold = Ant_args.accumulative_threshold
        self.temp_accu_reward = 0 
        
        # RP args (for online training)
        self.RP = None
        self.predict_with_action = Ant_args.predict_with_action

        if Ant_args.reward_pred or Ant_args.online_training:
            if Ant_args.online_training:
                load_path = None
            else:
                load_path = Ant_args.reward_pred

            obs_size = self.observation_space.sample().size
            action_size = self.action_space.sample().size
            fc_layer_size = (obs_size + action_size) if self.predict_with_action else obs_size

            self.RP = Predictor(threshold = 4,  # TODO: add to arguments?
                                device = Ant_args.device,
                                model_load_path = load_path,
                                env_name = "",
                                show_trajectory = None,
                                num_of_parallel_envs = 1,
                                siamese_type = " linear",
                                input_fc_layer_size = fc_layer_size,
                                save_results_path = Ant_args.log_dir)
            
            self.rp_factor = Ant_args.rp_factor

        self.online_training = Ant_args.online_training
        self.device = Ant_args.device

        if self.online_training:
            self.replay_size = Ant_args.replay_size
            self.obs_buf = []
            self.r_buf = []
            self.done_buf = []
            self.step_count = 0
            self.win_size = Ant_args.win_size
            self.train_interval = Ant_args.train_interval

       # End of RP args  

    def step(self, a):
        #TODO: this step is only for uniform decision distance experiment
        #uniform_step = np.random.uniform(-1,1,3)
        #ob, reward, done, _ = super(RPHopperEnv, self).step(uniform_step)

        prev_ob = self._get_obs()

        ob, reward, done, _ = super(RPAntEnv, self).step(a)
        if (not hasattr(self, 'nsteps')):
            return ob, reward, done, _

        self.nsteps += 1
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

        if done:
            info['episode'] = {'r': self.tot_reward, 'l': self.nsteps, 'sparse_r': self.tot_sparse_reward,
            'distance': self.sim.data.qpos[0] - self.abs_first_pos}
        #return ob, reward, done, info
        
        if not self.RP:
            return  ob, reward, done, info

        if self.online_training: 
            train_ob = np.concatenate((ob, a), axis=0)
            self._add_to_buf(train_ob, done, info)
            self.step_count += 1
            if self.step_count >= self.train_interval:
                self._train_rp()
                self.step_count = 0

        r_int, var = self.RP.predict(train_ob)
        r_unk = self._rp_var_reward(var)
        R = reward + self.rp_factor * (r_int + r_unk)

        return ob, R, done, info
  
    def _train_rp(self):

          # reshape data
          obs = np.array(self.obs_buf)
          dones = np.array(self.done_buf)
          rs = np.array(self.r_buf)
   
          # create windows
          train_obs = []
          acc_r = []
          for i in range(obs.shape[0] - self.win_size):
              if True in dones[i: i + self.win_size - 1]:
                  continue
              win = obs[i:i + self.win_size]
              train_obs.append(win)
              acc_r.append(np.sum(rs[i: i+self.win_size]))
          if not train_obs:
              # empty trainig batch
              return
              
          train_obs = np.array(train_obs)
          acc_r = np.array(acc_r)
          acc_r = acc_r.reshape((acc_r.shape[0],1))
          
          self.RP.one_batch_train(train_obs, acc_r, self.device, 4)

    def _rp_var_reward(self, var):
        # return reward depending on rp models disagreement
        return 0
     
    def _add_to_buf(self, ob, done, info):
        self.obs_buf.append(ob)
        self.done_buf.append(done)
        self.r_buf.append(info['special_r'])

        if len(self.obs_buf) > self.replay_size:
            self.obs_buf = self.obs_buf[1:]
            self.r_buf = self.r_buf[1:]
            self.done_buf = self.done_buf[1:]

    def reset_model(self):
        self.temp_accu_reward = 0
        self.tot_reward = 0
        self.tot_sparse_reward = 0
        self.nsteps = 0
        return super(RPAntEnv, self).reset_model()
