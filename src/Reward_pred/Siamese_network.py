import numpy as np
import gym
import torch
import torch.nn as nn

#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.sb3_extentions.custom_cnn2 import CustomCnn

#Reward predictor arch params
REWARD_NORM_FACTOR = 1

FIRST_FC_NN = 1024
SECOND_FC_NN = 256
NUM_OF_DIFF_MODELS = 3

BATCH_SIZE = 8
TRAJECTORY_LENGTH = 3
TRAJECTORY_TEST_LENGTH = 3
SAVE_EPISODES = 200
TEST_SET_LENGTH = 20

class Siam__conv_nn(CustomCnn):
    '''
        deep convolution network, representing Q function
        observation_shape - tuple of input shape
        outputs - size of outputs (should be 1 for reward predictor)
    '''

    def __init__(self, observation_shape, outputs, feature_dims, action_space=None, use_bn=False):
        observation_space = gym.spaces.Box(low=0, high=1, shape=observation_shape)
        super(Siam__conv_nn, self).__init__(observation_space, outputs, feature_dims=feature_dims,
                                            use_final_activation=False, use_bn=use_bn, action_space=action_space)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, trajectory, action=None, mask=None):
        '''
        forward propagation of the state/action values in the network - result is a one scalar of reward
        :param trajectory: tuple of lists contains (state,action) is a picture of the cartpole state
        :return: double, reward
        '''
        # If the obs is in batch mode (e.g several envs of PPO) keep it as batch output by
        # switching the first and second element's order of the input tensor (make batch second and window first)

        # TODO: implement using mask in cases of different window sizes

        if len(trajectory.shape) == 5:  # conv obs
            trajectory = trajectory.permute(1, 0, 2, 3, 4)

        if self._predict_with_action and len(trajectory.shape) == 3:
            action = action.permute(1, 0, 2)

        sum_of_reward = 0
        for i, state in enumerate(trajectory):
            feature = self.cnn(state)
            # If predict with action, concatenate action to extracted features
            if self._predict_with_action:
                feature = torch.cat((feature, action[i]), dim=1)
            sum_of_reward += self.linear(feature)
        return sum_of_reward

class Siam__FC_linear_nn(nn.Module):
    ''' deep FC network, representing Q function
        features_len - list of sizes of layers
        outputs - size of outputs (should be 1 for reward predictor)
    '''

    def __init__(self, features_len, hidden_layers_sizes, outputs, action_space=None,):
        super(Siam__FC_linear_nn, self).__init__()
        layers_list = []
        features_len = int(features_len)

        self._predict_with_action = action_space is not None
        if self._predict_with_action:
            features_len += action_space

        layers_list.append(nn.Linear(int(features_len), hidden_layers_sizes[0]))
        layers_list.append(nn.ReLU())
        for index in range(len(hidden_layers_sizes) - 1):
            layers_list.append(nn.Linear(hidden_layers_sizes[index], hidden_layers_sizes[index + 1]))
            layers_list.append(nn.ReLU())
        
        layers_list.append(nn.Linear(hidden_layers_sizes[-1], outputs))
        for layer in layers_list:
            weight_init(layer)
        self.MLP = nn.Sequential(*layers_list)
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, trajectory, action=None):
        '''
        forward propagation of the state/action values in the network - result is a one scalar of reward
        :param trajectory: tuple of lists contains (state,action) is a picture of the cartpole state
        :return: double, reward
        '''
        
        # If the obs is in batch mode (e.g several envs of PPO) keep it as batch output by
        # switching the first and second element's order of the input tensor (make batch second and window first)

        if len(trajectory.shape) == 2:  # Inference only
            # trajectory shape should be (N x S)
            # Where:
            #   N - number of envs during predict function
            #   S - number of Input size (Env observation size)
            # Output is a batch output of (Nx1) imidete rewards
            if self._predict_with_action:
                if len(action.shape) == 1:  # The shape of action is (N,) should be (N,1)
                    action = action.unsqueeze(axis=1)
                trajectory = torch.cat((trajectory, action), dim=1)
            sum_rewards = self.MLP(trajectory)
            return sum_rewards
        else:
        # trajectory shape should be (B X W X S)
        # Where:
        #   B - RP train batch size
        #   W - RP train window size
        #   S - number of Input size (Env observation size)
        # Output is a batch output of (Bx1) imidete rewards
        #   1) Should reshape input to be (BW X S)
        # 1.1) If predict with action, concatenate action to observation to be of shape (BE X [S+A])
        #   2) go through MLP to get rewards shape (BW X 1)
        #   3) Reshape Rewards to be (B X W)
        #   4) Sum along window axis to receive a total of (B X 1) rewards per batch
        #   Notice - make sure the reshaping is done well for each window!!!
            B = trajectory.shape[0]
            W = trajectory.shape[1]
            flat_trajectory = trajectory.view(B * W, -1)
            if self._predict_with_action:
                flat_action = action.view(B * W, -1)
                flat_trajectory = torch.cat((flat_trajectory, flat_action), dim=1)
            rewards = self.MLP(flat_trajectory)
            rewards = rewards.view(B, W)
            sum_rewards = torch.sum(rewards, dim=1, keepdim=True)
            return sum_rewards


def weight_init(m):
    '''
    This function randomize FC layer weights as gaussian variable
    :param m: layer of torch.nn type
    :return: No return value
    '''
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)