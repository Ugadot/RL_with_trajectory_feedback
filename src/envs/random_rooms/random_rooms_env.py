import numpy as np
from gym import error, core, spaces, utils
from gym.envs.registration import register
from gym.utils import seeding


class RandomRoomsEnv(core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rows=10, cols=10, upsample=1,
                 goal_in_state=True, n_inner_resets=1000,
                 inner_counter_in_state=True, max_steps=30,
                 goal_visible_in_room=False, cmd_args=None):

        # command line args
        self.rooms = cmd_args.rooms  # currently only 0 / 2 / 4 supported
        self.obs_type = cmd_args.obs_type
        self.flat_obs = (cmd_args.obs_type == 'flat')
        
        self.far_from_goal = cmd_args.far_from_goal
        # decides if to randomize on each reset
        self.keep_walls = cmd_args.const_rooms
        self.keep_goal = cmd_args.const_goal
        if self.keep_walls:
            self.walls_rng = np.random.RandomState(cmd_args.seed)
        else:
            self.walls_rng = np.random.RandomState()
        if self.keep_goal:
            self.goal_and_agent_rng = np.random.RandomState(cmd_args.seed)
        else:
            self.goal_and_agent_rng = np.random.RandomState()


        self.rows, self.cols = cmd_args.rows, cmd_args.cols
        self.max_steps = (self.rows + self.cols ) * cmd_args.max_steps_fact
        self.inner_counter_in_state = cmd_args.inner_counter_in_state
        self.n_inner_resets = cmd_args.n_inner_resets

        self.upsample = upsample
        self.goal_in_state = goal_in_state
        # self.n_inner_resets = n_inner_resets
        # self.inner_counter_in_state = inner_counter_in_state
        # self.max_steps = max_steps
        self.goal_visible_in_room = goal_visible_in_room

        n_channels = 2 + goal_in_state + self.inner_counter_in_state
        #print("n_channels: ", n_channels)
        self.action_space = spaces.Discrete(4)

        # decides if to randomize on each reset
        # self.keep_walls = True
        # self.keep_goal = True

        if self.obs_type == 'flat':
            self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(3 * self.rows * self.cols ,),
                                            dtype=np.float32)
        # Reduced obs space = (goal_x, goal_y, agent_x, agent_y) - Should be used only with const-rooms argument
        elif self.obs_type == 'reduced':
            self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
            assert cmd_args.seed != None, "can't use reduced observation without seed (walls must not change)"

        # Convulutional Obs space
        else:
            self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(n_channels, self.cols * self.upsample, self.rows * self.upsample),
                                            dtype=np.float32)

        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        # self.rng = np.random.RandomState(1234)

        self.map, self.seed = self._randomize_walls()
        self.goal_cell, self.goal = self._random_from_map()
        
        self.counter = 0
        self.hard_reset_counter = 0
        self.nsteps = 0
        self.tot_reward = 0

        self.visible_goal = False # if False, goal is visible only when player in the same room
        
        #Restart actor location
        if self.far_from_goal:
            self.state_cell, self.state = self._farthest_from_goal()
        else:
            self.state_cell, self.state = self._random_from_map()

        # debug
        print("\nnew env, seed ", cmd_args.seed)
        # print map nicely
        print_map = self.map.astype(int)
        print_map[self.goal_cell[0], self.goal_cell[1]] = 2
        print_map[self.state_cell[0], self.state_cell[1]] = 3
        back_map_modified = np.vectorize(get_color_coded_background)(print_map)
        print_a_ndarray(back_map_modified, row_sep="")
        # print_map = np.vectorize(get_color_coded_str)(print_map)
        # print("\n".join([" ".join(["{}"] * 10)] * 10).format(*[x for y in print_map.tolist() for x in y]))

    def reset(self, override=False, hard=False):

        self.counter += 1

        if (override and hard) or (not override and self.counter >= self.n_inner_resets):
            print('FINAL RESET with reward: {}'.format(self.tot_reward))
            if not self.keep_walls:
                self.map, self.seed = self._randomize_walls()
                print('RESET WALLS')
            if not self.keep_goal:
                self.goal_cell, self.goal = self._random_from_map()
                print('RESET GOAL')
            self.counter = 0
            self.hard_reset_counter += 1

        #Restart actor location
        if self.far_from_goal:
            self.state_cell, self.state = self._farthest_from_goal()
        else:
            self.state_cell, self.state = self._random_from_map()

        self.nsteps = 0
        self.tot_reward = 0

        obs = self._im_from_state()

        # return obs
        # hm - preprocess
        #return obs.T
        return obs

    def get_current_obs(self):
        obs = self._im_from_state()
        return obs

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        if (type(action) != type([]) and type(action) != type(np.ndarray([]))):
          action = [action]
        next_cell = self.state_cell + self.directions[action[0]]
        if self.map[next_cell[0] * self.upsample, next_cell[1] * self.upsample] == 0:
            self.state_cell = next_cell
            self.state = np.zeros_like(self.map)
            for i in [0]:#[-1, 0, 1]:
                for j in [0]:#[-1, 0, 1]:
                    self.state[(self.state_cell[0] + i) * self.upsample:
                               (self.state_cell[0] + i + 1) * self.upsample,
                                (self.state_cell[1] + j) * self.upsample:
                                (self.state_cell[1] + j + 1) * self.upsample] = 1

        done = np.all(self.state_cell == self.goal_cell)
        obs = self._im_from_state()
        r = float(done)  

        if self.nsteps >= self.max_steps:
            done = True

        self.tot_reward += r
        
        self.nsteps += 1
        info = dict()
        info['state_cell'] = self.state_cell  # Added to info inorder to help Dijk Wrapper
        info['step'] = {'inner_counter': self.counter,
                        'hard_reset_counter': self.hard_reset_counter}

        if 'wandb_log' not in info.keys():
            info['wandb_log'] = {}
        info['wandb_log']['sparse_reward'] = r
        info['wandb_log']['done'] = 0

        # Log distance from goal in the current step
        info['wandb_log']['L1_distance_from_goal'] = np.linalg.norm(self.state_cell - self.goal_cell) 
        info['wandb_log']['hard_reset_counter'] = self.hard_reset_counter

        if done:
            info['episode'] = {'r': self.tot_reward, 'l': self.nsteps}
            info['wandb_log']['done'] = 1

        return obs, r, done, info


    def _random_from_map(self, far_from_goal = False):
        cell = self.goal_and_agent_rng.choice(self.rows), self.goal_and_agent_rng.choice(self.cols)
        while (self.map[cell[0] * self.upsample, cell[1] * self.upsample] == 1):
            cell = self.goal_and_agent_rng.choice(self.rows), self.goal_and_agent_rng.choice(self.cols)
        
        map = np.zeros_like(self.map)
        for i in [0]:#[-1, 0, 1]:
            for j in [0]:#[-1, 0, 1]:
                map[(cell[0] + i) * self.upsample:
                    (cell[0] + i + 1) * self.upsample,
                    (cell[1] + j) * self.upsample:
                    (cell[1] + j + 1) * self.upsample] = 1

        return np.array(cell), map


    def _farthest_from_goal(self):
        cells = [(1,1), (self.rows - 2 , 1), (1, self.cols -2), (self.rows - 2, self.cols - 2)]
        max_distance = 0
        max_cell = (-1,-1)
        for cell in cells:
            distance = np.linalg.norm(cell[0] - self.goal_cell[0]) +  np.linalg.norm(cell[1] - self.goal_cell[1])
            if (distance > max_distance):
                max_cell = cell
                max_distance = distance
        cell = max_cell
        
        map = np.zeros_like(self.map)
        for i in [0]:#[-1, 0, 1]:
            for j in [0]:#[-1, 0, 1]:
                map[(cell[0] + i) * self.upsample:
                    (cell[0] + i + 1) * self.upsample,
                    (cell[1] + j) * self.upsample:
                    (cell[1] + j + 1) * self.upsample] = 1

        return np.array(cell), map


    def _im_from_state(self):
        im_list = [self.state, self.map]
        if self.goal_in_state:
            if self.goal_visible_in_room:
                if self._which_room(self.state_cell) == self._which_room(self.goal_cell):
                    im_list.append(self.goal)
                else:
                    im_list.append(np.zeros_like(self.map))
            else:
                im_list.append(self.goal)
        if self.inner_counter_in_state:
            im_list.append((self.counter / self.n_inner_resets) * np.ones_like(self.map))

        if self.obs_type == 'original':
            obs = np.stack(im_list, axis=-1)
            return obs.T

        elif self.obs_type == 'flat':
        # Concatenate each channel as one flat tensor
            flat_maps = [(down_sample_map(m, self.upsample)).flatten() for m in im_list]
            flat_obs = np.concatenate(tuple(flat_maps), 0)
            return flat_obs
            # flat_map = (down_sample_map(self.map, self.upsample)).flatten()
            # flat_goal = (down_sample_map(self.goal, self.upsample)).flatten()
            # flat_state = (down_sample_map(self.state, self.upsample)).flatten()
            # flat_obs = np.concatenate((flat_map, flat_goal, flat_state), 0)
            # return flat_obs

        elif self.obs_type == 'reduced':
            player = im_list[0]
            goal = im_list[2]
            player_place = np.where(player == 1.0)
            goal_place = np.where(goal == 1.0)

            # Normalize (x,y) so that the values would be between [0,1]
            player_place = (player_place[0] / self.rows, player_place[1] / self.cols)
            goal_place = (goal_place[0] / self.rows, goal_place[1] / self.cols)
            return np.concatenate(player_place + goal_place)

        else:
            # Should not reach this point
            assert(False)

    def _which_room(self, cell):
        if cell[0] <= self.seed[0] and cell[1] <= self.seed[1]:
            return 0
        elif cell[0] <= self.seed[0] and cell[1] > self.seed[1]:
            return 1
        elif cell[0] > self.seed[0] and cell[1] <= self.seed[1]:
            return 2
        else:
            return 3


    def _randomize_walls(self):
        map = np.zeros((self.rows * self.upsample, self.cols * self.upsample))
        map[0:self.upsample, :] = 1
        map[:, 0:self.upsample] = 1
        map[-self.upsample:, :] = 1
        map[:, -self.upsample:] = 1

        seed = (self.walls_rng.randint(3, self.rows - 3), self.walls_rng.randint(3, self.cols - 3))
        if self.rooms == 2:
            map[seed[0]*self.upsample:(seed[0] + 1)*self.upsample, :] = 1
            door3 = self.walls_rng.randint(1, seed[1])
            map[seed[0]*self.upsample:(seed[0] + 1)*self.upsample, door3*self.upsample:(door3+1)*self.upsample] = 0

        elif self.rooms == 4:
            # create rooms
            map[seed[0]*self.upsample:(seed[0] + 1)*self.upsample, :] = 1
            map[:, seed[1]*self.upsample:(seed[1] + 1)*self.upsample] = 1
            door1 = self.walls_rng.randint(1, seed[0])
            door2 = self.walls_rng.randint(seed[0] + 1, self.rows - 1)
            map[door1*self.upsample:(door1+1)*self.upsample, seed[1]*self.upsample:(seed[1] + 1)*self.upsample] = 0
            map[door2*self.upsample:(door2+1)*self.upsample, seed[1]*self.upsample:(seed[1] + 1)*self.upsample] = 0
            door3 = self.walls_rng.randint(1, seed[1])
            door4 = self.walls_rng.randint(seed[1] + 1, self.cols - 1)
            map[seed[0]*self.upsample:(seed[0] + 1)*self.upsample, door3*self.upsample:(door3+1)*self.upsample] = 0
            map[seed[0]*self.upsample:(seed[0] + 1)*self.upsample, door4*self.upsample:(door4+1)*self.upsample] = 0

        elif self.rooms == 9:
            assert self.rows > 10 or self.cols > 10
            # walls cols & rows
            seed1 = (self.walls_rng.randint(2, (self.rows // 2) - 1), self.walls_rng.randint(2, (self.cols // 2) - 1))
            seed2 = (self.walls_rng.randint((self.rows // 2)+ 1, self.rows - 2), self.walls_rng.randint((self.cols // 2) + 1, self.cols -2))

            # put walls in map
            map[seed1[0]*self.upsample:(seed1[0] + 1)*self.upsample, :] = 1
            map[:, seed1[1]*self.upsample:(seed1[1] + 1)*self.upsample] = 1
            map[seed2[0] * self.upsample:(seed2[0] + 1) * self.upsample, :] = 1
            map[:, seed2[1] * self.upsample:(seed2[1] + 1) * self.upsample] = 1

            # create doors
            doors1 = [self.walls_rng.randint(1, seed1[0]),
                      self.walls_rng.randint(seed1[0] + 1, seed2[0]),
                      self.walls_rng.randint(seed2[0]+ 1, self.rows - 1)]

            doors2 = [self.walls_rng.randint(1, seed1[1]),
                      self.walls_rng.randint(seed1[1] + 1, seed2[1]),
                      self.walls_rng.randint(seed2[1]+ 1, self.rows - 1)]
            # put doors in map
            for door in doors1:
                map[door*self.upsample:(door+1)*self.upsample, seed1[1]*self.upsample:(seed1[1] + 1)*self.upsample] = 0
                map[door * self.upsample:(door + 1) * self.upsample, seed2[1] * self.upsample:(seed2[1] + 1) * self.upsample] = 0

            for door in doors2:
                map[seed1[0] * self.upsample:(seed1[0] + 1) * self.upsample,door * self.upsample:(door + 1) * self.upsample] = 0
                map[seed2[0] * self.upsample:(seed2[0] + 1) * self.upsample,door * self.upsample:(door + 1) * self.upsample] = 0

        return map, seed

    def render(self, mode='human', close=False):
        '''
        creates and returns RGB images from  current state observation
        :param mode:
        :param close:
        :return:
        '''
        import matplotlib.pyplot as plt

        #obs.T = self._im_from_state()
        walls = obs[1,:,:]
        color_walls_t = np.array([walls,walls,walls])
        color_walls = np.ones_like(color_walls_t) - color_walls_t
        player = obs[0,:,:]
        color_player = np.array([np.zeros_like(player), player, player]) * -1
        goal = obs[2,:,:]
        color_goal = np.array([goal, goal, np.zeros_like(goal)]) * -1
        color_obs = color_player + color_walls + color_goal
        color_obs_im = color_obs.T
        plt.imshow(color_obs_im);
        plt.show(block=True)

        return color_obs_im

    def down_sample_map(self):
        simple_map = np.zeros([self.rows, self.cols])
        for i in range(self.rows):
            for j in range(self.cols):
                simple_map[i,j] = self.map[i * self.upsample, j * self.upsample]

        return simple_map

    def get_current_img(self):
        im_list = [self.state, self.map]
        if self.goal_in_state:
            if self.goal_visible_in_room:
                if self._which_room(self.state_cell) == self._which_room(self.goal_cell):
                    im_list.append(self.goal)
                else:
                    im_list.append(np.zeros_like(self.map))
            else:
                im_list.append(self.goal)
        if self.inner_counter_in_state:
            im_list.append((self.counter / self.n_inner_resets) * np.ones_like(self.map))


        obs = np.stack(im_list, axis=-1)
        obs = obs.T
        return get_img_from_ob(obs)

def get_img_from_ob(obs):
    '''
    creates and returns RGB images from given state observation
    :input param: state nd_array
    :return: colored image (not upsampled though)
    '''

    walls = obs[1,:,:]
    color_walls_t = np.array([walls,walls,walls])
    color_walls = np.ones_like(color_walls_t) - color_walls_t
    player = obs[0,:,:]
    color_player = np.array([np.zeros_like(player), player, player]) * -1
    goal = obs[2,:,:]
    color_goal = np.array([goal, goal, np.zeros_like(goal)]) * -1
    color_obs = color_player + color_walls + color_goal
    color_obs_im = color_obs.T
    color_obs_im = color_obs_im * 255.0
    return color_obs_im

def down_sample_map(map, scale):
    rows = int(map.shape[0] /scale)
    cols = int(map.shape[1] / scale)
    # print(rows, cols)
    simple_map = np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            simple_map[i,j] = map[i * scale, j * scale]

    return simple_map

def example():
    # env = RandomRoomsEnv(rows=5, cols=6, upsample=120)
    import argparse
    from rooms_args import get_random_rooms_arg_parser
    random_rooms_parser = get_random_rooms_arg_parser()
    parser = argparse.ArgumentParser(parents=[random_rooms_parser])
    parsed_args = parser.parse_args()

    env = RandomRoomsEnv(cmd_args=parsed_args)
    print("example created env")
    print(env.dijkstra_map)
    for i in range(10):
        action = i % 4
        print("step {}".format(i))
        obs, r, done, info = env.step(action)
        print("shape: ", obs.shape)
        print(env.state_cell)
        print(r)

def arrays_equal(a, b):
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True

# print map nicely
def get_color_coded_str(i):
    return "\033[3{}m{}\033[0m".format(i+1, i)

def get_color_coded_background(i):
    text_color = "3"
    if (i==0):  # Background
      text = " "
      bg_color = "7"
    elif (i==1):  # Walls
      text = " "
      bg_color = "7"
      return "\033[48;2;{};{};{}m{} \033[0m".format(153, 76, 0, " ")
    elif (i==3):  # Player
      text = "A"
      bg_color = "4"
    elif (i==2):  # Goal
      text = "G"
      bg_color = "2"
      text_color = "1"
      return "\033[4{}m\033[30m{} \033[0m".format(bg_color, text)
    return "\033[4{}m{} \033[0m".format(bg_color, text)

def print_a_ndarray(map, row_sep=" "):
    n, m = map.shape
    fmt_str = "\n".join([row_sep.join(["{}"]*m)]*n)
    print(fmt_str.format(*map.ravel()))
    print("\n")


if __name__ == '__main__':
    # example()
    import argparse
    from rooms_args import get_random_rooms_arg_parser
    random_rooms_parser = get_random_rooms_arg_parser()
    parser = argparse.ArgumentParser(parents=[random_rooms_parser])
    parsed_args = parser.parse_args()
    parsed_args.rooms = 9
    parsed_args.rows=20
    parsed_args.cols=20
    env = RandomRoomsEnv(cmd_args=parsed_args)

    import matplotlib.pyplot as plt
    obs = env.reset()
    plt.imshow(obs.T);
    plt.show()