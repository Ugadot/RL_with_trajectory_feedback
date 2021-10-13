import gym
import numpy as np

class DijkRewardWrapper(gym.Wrapper):
    """
      Wraps specifically rooms env to allow a Dijkstra reward calculations

      Writes the Dijk reward value to info['Dijk_reward']
    """

    def __init__(self, env, args):
        '''
        relevant args are:
            * dijk_power - Integer to power all DIjkstra values
            * dijk_lambda - Lambda factor to multiply all dijkstra calculated values
            * use_dijk_reward - Bool argument to whether replace the env reward with the Dijk reward or not
        '''
        super().__init__(env)
        
        # accu reward vars
        self.dijk_power = args.dijk_power  # 'liner' / 'squared'
        self.dijk_lambda = args.dijk_lambda
        self.use_dijkstra_reward = args.use_dijk_reward

        self.dijkstra_map = self.set_dijkstra_map(self.env)
        # float_formatter = "{:.2f}".format
        # np.set_printoptions(formatter={'float_kind':float_formatter})
        # print(self.dijkstra_map)
        
    def reset(self, override=False, hard=False):
        new_obs = self.env.reset(override=False, hard=False)
        self.dijkstra_map = self.set_dijkstra_map(self.env)
        return new_obs


    def step(self, action):
        observation, reward, done, info = super(DijkRewardWrapper, self).step(action)
        state_cell = info['state_cell']
        dijk_r = -1 * self.dijkstra_map[state_cell[0], state_cell[1]]
        
        info['rp_learned_reward'] = dijk_r
        if 'wandb_log' not in info.keys():
            info['wandb_log'] = {}
        info['wandb_log']['dijk_reward'] = dijk_r

        if self.use_dijkstra_reward:
            reward += self.dijk_lambda * dijk_r

        return observation, reward, done, info
        
        
    def set_dijkstra_map(self, env):
        max_goal_dist = 0
        simple_map = env.down_sample_map()
        dijk_max = env.rows * env.cols
        dijkstra_map = np.ones_like(simple_map) * dijk_max
        dijkstra_map[env.goal_cell[0], env.goal_cell[1]] = 0
        ind_list = [env.goal_cell]
        while len(ind_list) > 0:
            idx = np.array(ind_list.pop(0))

            neighbors = [
            idx + np.array([-1,0]),
            idx + np.array([1, 0]),
            idx + np.array([0, -1]),
            idx + np.array([0, 1])
            ]

            for n in neighbors:
                if n[0] < 0 or n[0] >= env.rows or n[1] < 0 or n[1] >= env.cols:
                    pass
                elif simple_map[n[0], n[1]] == 1:
                    pass
                elif dijkstra_map[n[0], n[1]] <= dijkstra_map[idx[0], idx[1]] + 1:
                    pass
                else:
                    dist = dijkstra_map[idx[0], idx[1]] + 1
                    dijkstra_map[n[0], n[1]] = dist
                    if dist > max_goal_dist:
                        max_goal_dist = dist
                    ind_list.append(n)

        dijkstra_map = dijkstra_map * (1/max_goal_dist)

        if self.dijk_power > 1:
            dijkstra_map = np.power(dijkstra_map, self.dijk_power)

        return dijkstra_map

    @staticmethod
    def get_args(parser):
         #Dijkstra argument
        Dijk_group = parser.add_argument_group("Dijkstra args")
        Dijk_group.add_argument('--use_dijk_reward', default=False, action='store_true', help='Use Dijk Wrapper, or not needed')
        Dijk_group.add_argument('--dijk_power', default=1, type=int, help='power factor to the dijk reward')
        Dijk_group.add_argument('--dijk_lambda', default=0.1, type=float, help='multiply factor for dijk reward')
        return parser
