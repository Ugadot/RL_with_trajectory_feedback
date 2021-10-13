import gym
import numpy as np
import matplotlib.pyplot as plt


class PongDenseWrapper(gym.Wrapper):
    """
      Wraps specifically rooms env to allow a Dijkstra reward calculations

      Writes the Dijk reward value to info['Dijk_reward']
    """

    def __init__(self, env, args):
        '''
        relevant args are:
            * dense_lambda - Lambda factor to multiply all dense calculated values
            * use_dense_reward - Bool argument to whether replace the env reward with the dense reward or not
        '''
        super().__init__(env)
        self.dense_lambda = args.dense_lambda
        self.use_dense_reward = args.use_dense_reward

        # preprocess obs space
        self._frame_stack = args.frame_stack
        if self._frame_stack == 1:
          self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 80, 80), dtype=np.float32)
        else:
          self.observation_space = gym.spaces.Box(low=0, high=1, shape=(80, 80), dtype=np.float32)

        self.dense_type = args.dense_type

    def reset(self):
        new_obs = self.env.reset()
        img = self._preprocess_obs(new_obs)
        self.score = 0
        return img

    def _rgb2gray(self, rgb):
      r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
      gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32) / 255.0
      if self._frame_stack == 1:
        gray =  np.expand_dims(gray, axis=2)
      gray = np.moveaxis(gray, -1, 0)
      return gray

    def _preprocess_obs(self, image):
        '''
        Returns an upsamled normalized image in the size of 60x60
        '''
        bkg_color = np.array([144, 72, 17])
        img = image[34:-16:2, ::2]
        gs_img = self._rgb2gray(img)
        return gs_img

    def step(self, action):
        observation, reward, done, info = super(PongDenseWrapper, self).step(action)
        observation = self._preprocess_obs(observation)

        self.score += reward
        if self.dense_type == 1:
            dense_reward = self.find_vertical_distance_reward(get_image(self.unwrapped.ale))
        elif self.dense_type == 2:
            dense_reward = self.find_dense_reward(get_image(self.unwrapped.ale))
        else:
            dense_reward = 0
        # print("dense r: ", dense_reward)
        ram_obs = to_ram(self.unwrapped.ale)
        info['rp_learned_reward'] = dense_reward
        if 'wandb_log' not in info.keys():
            info['wandb_log'] = {}
        info['wandb_log']['dense_reward'] = dense_reward
        info['wandb_log']['sparse_reward'] = reward
        info['wandb_log']['score'] = self.score

        if self.use_dense_reward:
            #reward += self.dense_lambda * dense_reward
            reward = self.dense_lambda * dense_reward
        # self.env.render()
        return observation, reward, done, info

    def find_vertical_distance_reward(self, obs):
        # relevent part of image
        ball_region = obs[34:194, :,:]
        ball_color = (236,236,236)
        player_region = obs[34:194, 140:141, :]
        player_color = (92,186,92)
        max_dist = 160

        ball = np.where(np.all(ball_region == ball_color, axis=-1))
        # print(ball)
        if len(ball[0]) == 0:
            # ball is not seen
            return 0
        ball_top = ball[0][0]

        player = np.where(np.all(player_region == player_color, axis=-1))
        # print(ball)
        player_len = len(player[0])
        center_idx = 6
        if player_len == 0:
            return 0
        if player_len < 7:
            center_idx = player_len // 2
        player_center = player[0][center_idx]

        dist = np.abs(player_center - ball_top)

        dist_reward =  1 - (dist / max_dist)
        return dist_reward

    def find_dense_reward(self, obs):
        # relevent part of image
        ball_region = obs[34:194, :, :]
        ball_color = (236, 236, 236)
        player_region = obs[34:194, 140:141, :]
        player_color = (92, 186, 92)
        opponent_region = obs[34:194, 18:19, :]
        opponent_color = (213, 130, 74)

        max_dist = 160

        ball = np.where(np.all(ball_region == ball_color, axis=-1))
        # print(ball)
        if len(ball[0]) == 0:
            # ball is not seen
            return 0
        ball_top = ball[0][0]
        ball_hor_pos = ball[1][0]

        player = np.where(np.all(player_region == player_color, axis=-1))
        # print(ball)
        player_len = len(player[0])
        center_idx = 6
        if player_len ==0:
            return 0
        if player_len < 7:
            center_idx = player_len // 2
        player_center = player[0][center_idx]

        opponent = np.where(np.all(opponent_region == opponent_color, axis=-1))
        opponenet_len = len(opponent[0])
        opp_center_idx = 6
        if opponenet_len == 0:
            return 0
        if opponenet_len < 7:
            opp_center_idx = opponenet_len // 2
        opp_center = opponent[0][opp_center_idx]

        # calc reward
        if ball_hor_pos > 100:
            # closer to player - check player is close
            dist = np.abs(player_center - ball_top)
            dist_reward = 1 - (dist / max_dist)
        elif ball_hor_pos < 60:
        # closer to opponent - check opp is far from ball
            dist = np.abs(opp_center - ball_top)
            dist_reward = (dist / max_dist)
        else:
            dist_reward = 0
        return dist_reward


    @staticmethod
    def get_args(parser):
        Dense_group = parser.add_argument_group("Dense Pong args")
        Dense_group.add_argument('--use_dense_reward', default=False,
                                action='store_true',
                                help='Use Dense Pong Wrapper, or not needed')
        Dense_group.add_argument('--dense_lambda', default=0.1, type=float,
                                help='multiply factor for dense reward')
        Dense_group.add_argument('--frame_stack', default=1, type=int,
                                help='number of frames to stack [default=1]')
        Dense_group.add_argument('--dense_type', default=2, type=int,
                                 help='dense type')

        return parser

    def get_current_img(self):
        return get_image(self.ale)

def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size), dtype=np.uint8)
    ale.getRAM(ram)
    return ram

def get_image(ale):
    return ale.getScreenRGB2()

