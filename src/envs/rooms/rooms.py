import numpy as np
from gym import core, spaces


class RoomsEnv(core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rows=15, cols=15,
                 goal_in_state=True, max_steps=None,
                 goal_only_visible_in_room=False, seed=None):
        self.rows, self.cols = rows, cols
        if max_steps is None:
            self.max_steps = 3 * (rows + cols)
        else:
            self.max_steps = max_steps

        self.goal_in_state = goal_in_state

        self.goal_only_visible_in_room = goal_only_visible_in_room

        n_channels = 2 + goal_in_state
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_channels, self.rows, self.cols), dtype=np.float32)
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self.map, self.seed = self._randomize_walls()
        self.goal_cell, self.goal = self._random_from_map()
        self.state_cell, self.state = self._random_from_map()

        self.tot_reward = 0

    def reset(self):
        self.state_cell, self.state = self._random_from_map()

        self.nsteps = 0
        self.tot_reward = 0

        obs = self._im_from_state()

        return obs

    def step(self, action: int):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        next_cell = self.state_cell + self.directions[action]
        if self.map[next_cell[0], next_cell[1]] == 0:
            self.state_cell = next_cell
            self.state = np.zeros_like(self.map)
            for i in [0]:#[-1, 0, 1]:
                for j in [0]:#[-1, 0, 1]:
                    self.state[(self.state_cell[0] + i):
                               (self.state_cell[0] + i + 1),
                                (self.state_cell[1] + j):
                                (self.state_cell[1] + j + 1)] = 1

        done = np.all(self.state_cell == self.goal_cell)
        obs = self._im_from_state()
        r = float(done)

        if self.nsteps >= self.max_steps:
            done = True

        self.tot_reward += r
        self.nsteps += 1
        info = dict()

        if done:
            info['episode'] = {'r': self.tot_reward, 'l': self.nsteps}

        return obs, r, done, info

    def _random_from_map(self):
        cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
        while self.map[cell[0], cell[1]] == 1:
            cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
        map = np.zeros_like(self.map)
        for i in [0]:#[-1, 0, 1]:
            for j in [0]:#[-1, 0, 1]:
                map[(cell[0] + i):
                    (cell[0] + i + 1),
                    (cell[1] + j):
                    (cell[1] + j + 1)] = 1

        return np.array(cell), map

    def _im_from_state(self):
        im_list = [self.state, self.map]
        if self.goal_in_state:
            if self.goal_only_visible_in_room:
                if self._which_room(self.state_cell) == self._which_room(self.goal_cell):
                    im_list.append(self.goal)
                else:
                    im_list.append(np.zeros_like(self.map))
            else:
                im_list.append(self.goal)

        return np.stack(im_list, axis=0).astype(np.int8)

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
        map = np.zeros((self.rows, self.cols))
        map[0, :] = 1
        map[:, 0] = 1
        map[-1:, :] = 1
        map[:, -1:] = 1

        seed = (self.rng.randint(2, self.rows - 2), self.rng.randint(2, self.cols - 2))

        map[seed[0]:seed[0] + 1, :] = 1
        map[:, seed[1]:(seed[1] + 1)] = 1
        door1 = self.rng.randint(1, seed[0])
        door2 = self.rng.randint(seed[0] + 1, self.rows - 1)
        map[door1:(door1+1), seed[1]:(seed[1] + 1)] = 0
        map[door2:(door2+1), seed[1]:(seed[1] + 1)] = 0
        door3 = self.rng.randint(1, seed[1])
        door4 = self.rng.randint(seed[1] + 1, self.cols - 1)
        map[seed[0]:(seed[0] + 1), door3:(door3+1)] = 0
        map[seed[0]:(seed[0] + 1), door4:(door4+1)] = 0

        return map, seed

    def render(self, mode='human', close=False):
        return 0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = RoomsEnv()
    obs = env.reset()
    plt.imshow(obs);
    plt.show()