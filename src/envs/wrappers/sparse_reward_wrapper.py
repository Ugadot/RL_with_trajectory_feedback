import gym

class SparseRewardWrapper(gym.Wrapper):
    """
      Wraps the environment to allow a Sparse reward calculations
      Currently implemented only:
        * Accumulative rewards
      
      Writes the Sparse reward value to info['step_sparse_reward']
    """

    def __init__(self, env, args):
    '''
      relevant args are:
        * sparse_reward_value - Value of the Sparse reward when used
        * accumulative_threshold - Threshold of Accumulative reward, when total episode reward reaches this treshold we return the Sparse value
        * use_sparse_reward - Bool argument to whether use the spare reward or not
    '''
        super(SparseRewardWrapper).__init__(env, args)
        
        # accu reward vars
        self.sparse_reward_value = args.sparse_reward_value
        self.accumulative_threshold = args.accumulative_threshold
        self.use_sparse_reward = args.use_sparse_reward
        self.temp_accu_reward = 0 


    def reset(self, **kwargs):
        self.tot_sparse_reward = 0
        self.temp_accu_reward = 0
        return self.env.reset(**kwargs)


    def step(self, action):
        observation, reward, done, info = super(SparseRewardWrapper, self).step(action)

        self.temp_accu_reward  += reward

        # Calc accu_reward
        sparse_reward = 0
        if (self.temp_accu_reward >= self.accumulative_threshold):
            # zero acc when reach threshold
            sparse_reward = self.sparse_reward_value
            self.temp_accu_reward = 0
        if (self.use_sparse_reward):
            reward = sparse_reward
        self.tot_sparse_reward += sparse_reward       
        
        if 'wandb_log' not in info.keys():
            info['wandb_log'] = {}
        info['wandb_log']['sparse_episode_reward'] = self.tot_sparse_reward
        info['wandb_log'][['sparse_reward'] = self.sparse_reward
        
        return observation, reward, done, info