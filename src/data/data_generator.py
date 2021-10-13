import numpy as np
import h5py
import os
from stable_baselines3.common.vec_env import VecEnv


class DataGenerator:
    def __init__(self, env: VecEnv, agent, info_keys=[], generate_images=False):
        self.env = env
        self.agent = agent
        self.info_keys = info_keys # info keys to save in data
        self.data = dict(states=[], next_state=[], actions=[], rewards=[], dones=[])
        for k in info_keys:
            self.data[k] = []
        if generate_images:
            self.data['images'] = []
        #
        # if hasattr(env, 'get_user_data'):
        #     self.data['user_data'] = []

        if hasattr(env, 'get_user_data'):
            self.data['user_data'] = []

        self.current_state = self.env.reset()
        #self.data['states'].append(np.copy(self.current_state).astype(np.int8))

    def update(self, env, agent):
        self.env = env
        self.agent = agent

    def step(self, deterministic=False):
        action = self.agent.predict(self.current_state, deterministic=deterministic)
        if type(action) == tuple:
            action = action[0]

        next_state, reward, done, info = self.env.step(action)
        # No need to reset on done, as env is of type VecEnv

        self.data['states'].append(self.current_state)
        self.data['next_state'].append(next_state)
        self.data['rewards'].append(reward)
        self.data['actions'].append(action)
        self.data['dones'].append(done)
        if 'images' in self.data.keys():
            #
            # im = self.env.envs[0].env.sim.render(420,380)
            # self.data['images'].append(np.expand_dims(im, 0))
            if isinstance(self.env, VecEnv):
                imgs = self.env.env_method("get_current_img")
            else:
                assert False, 'save images not supported inun vectorized env'
            if type(imgs) == list:
                imgs = np.stack(imgs)
            self.data['images'].append(imgs)

        if 'user_data' in self.data.keys():
            user_data = self.env.get_user_data(next_state, reward, done, info)
            if type(user_data) == list:
                user_data = np.stack(user_data)
            self.data['user_data'].append(user_data)


        for k in self.info_keys:
            if type(info) == list or type(info) == tuple:
                self.data[k].append([i[k] for i in info])
            else:
                self.data[k].append(info[k])
        self.current_state = np.copy(next_state)

    def save(self, path, save_idx=None):
        #assert len(self.data['states']) != len(self.data['rewards']), "data lists sizes mismatch"
        if save_idx is None:
            file_path = os.path.join(path, 'data.h5')
        else:
            file_path = os.path.join(path, f'data_{save_idx}.h5')
        hf = h5py.File(file_path, 'w')
        for k, v in self.data.items():
            v_numpy = np.array(v)
            print("key: {} shape: {}".format(k, v_numpy.shape))
            data_to_save = np.reshape(v, (np.prod(v_numpy.shape[0:2]),) + v_numpy.shape[2:], order='F')
            hf.create_dataset(k, data=data_to_save)
        hf.close()
        return self.data
