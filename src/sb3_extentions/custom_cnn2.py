import gym
import torch
from torch import nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space


class CustomCnn(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, output_dim: int = 512, feature_dims=[],
                 use_final_activation=True, use_bn=False, action_space=None):  # TODO: rename use_final_activation to RL usage
        super(CustomCnn, self).__init__(observation_space, output_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        input_dim = observation_space.shape[1]
        kernel_sizes = [max(input_dim // 5, 1), max(input_dim // 7, 1), max(input_dim // 11, 1)]
        strides = [max(input_dim // 16, 1), max(input_dim // 32, 1), 1]
        filters = [256 // kernel_sizes[0], 256 // kernel_sizes[1], 256 // kernel_sizes[2]]

        cnn_layers = []
        for index, (in_c, out_c) in enumerate(zip([n_input_channels] + filters[:-1], filters)):
            cnn_layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_sizes[index], stride=strides[index], padding=0))
            if use_bn:
                cnn_layers.append(nn.BatchNorm2d(filters[index]))
            cnn_layers.append(nn.ReLU())
        
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # If added action space than it should be added to the linear part
        self._predict_with_action = action_space is not None
        if action_space:
            n_flatten += action_space

        linear_layers = []
        for in_s, out_s in zip([n_flatten] + feature_dims, feature_dims + [output_dim]):
            linear_layers.append(nn.Linear(in_s, out_s))
            linear_layers.append(nn.ReLU())
        if not use_final_activation:
            linear_layers = linear_layers[:-1]
        
        self.linear = nn.Sequential(*linear_layers)

        print('Creating custom CNN arch with the following parameters:')
        print(self.cnn)
        print(self.linear)

        # clip output
        # TODO: add to args
        self.clip = True
        self.htanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)

        
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.cnn(observations))
        if self.clip:
            out = self.htanh(x)
        return out
