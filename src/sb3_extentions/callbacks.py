import os
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from abc import abstractmethod
from src.data.data_generator import DataGenerator
from src.utils.user_interface import saved_data_render
from multiprocessing import Process

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process
from tqdm import tqdm
import wandb

class SaveCallback(BaseCallback):
    def __init__(self, save_path,  verbose=0):
        super(SaveCallback, self).__init__(verbose)
        self.save_path = save_path
        self.counter = 1

    def _init_callback(self) -> None:
        env_name = self.training_env.get_attr('spec')[0].id
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _on_step(self) -> bool:
        return True


class SaveModelCallback(SaveCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveModelCallback, self).__init__(save_path=save_path, verbose=verbose)

    def _on_step(self) -> bool:
        print(f'Saving model {self.counter}')
        model_save_path = os.path.join(self.save_path, f'model_{self.counter}')
        self.model.save(model_save_path)
        self.counter += 1
        return True


class SaveDataCallback(SaveCallback):
    def __init__(self, save_path, data_size=1000000, info_keys=[], save_images=False, show_images=False, verbose=0):
        super(SaveDataCallback, self).__init__(save_path=save_path, verbose=verbose)
        self.data_size = data_size
        self.info_keys = info_keys
        self.save_images = save_images
        self.show_images = show_images
        self.show_process = None

    def _on_step(self) -> bool:
        print(f'Generating data {self.counter}')
        data_generator = DataGenerator(env=self.training_env, agent=self.model, info_keys=self.info_keys,
                                       generate_images=self.save_images or self.show_images)

        for _ in tqdm(range(self.data_size // self.training_env.num_envs)):
            data_generator.step(deterministic=False)

        saved_data = data_generator.save(self.save_path, self.counter)
        if 'images' in saved_data.keys() and self.show_images:
            print("show recorded data")
            if self.show_process:
                if self.show_process.is_alive():
                    print("kill old show process")
                    self.show_process.kill()

            self.show_process = Process(target=saved_data_render, args=(np.array(saved_data['images']),))

            print("start show process")
            self.show_process.start()
            # saved_data_render(saved_data['images'])
        self.counter += 1
        return True

class RenderCallback(BaseCallback):
    def __init__(self, create_env=None, sim_len=5000, verbose=1, deterministic=False):
        super(RenderCallback, self).__init__(verbose=verbose)
        self.sim_len = sim_len
        self.deterministic = deterministic
        self.create_env = create_env
        
    def _on_step(self) -> bool:
        print("running simulation")
        if self.create_env:
            print("creating new env")
            env = self.create_env()
        else:
            print("using train env")
            env=self.training_env
        agent=self.model

        state = env.reset()
        print("iterating on env")
        for i in tqdm(range(self.sim_len)):
            env.render()
            action = agent.predict(state, deterministic=self.deterministic)
            if type(action) == tuple:
                action = action[0]
                state, r, done, i = env.step(action)
                if done:
                    env.reset()

        if self.create_env:
            print("closing env")
            env.close()    
        return True



class ShowTrajectoryCallback(BaseCallback):
    def __init__(self, show_func, show_interval=1, verbose=0):
        super(ShowTrajectoryCallback, self).__init__(verbose)
        self.show_interval = show_interval
        self.show_func = show_func
        self.counter = 1

    def _on_step(self) -> bool:
        print("show trajectory")
        print(self.model.rollout_buffer)
        obs = self.model.rollout_buffer.observations
        # obs.shape: [traj_len, n_envs, env_obs.shape]
        print(type(obs))
        #print(obs)
        print(obs.shape)
        permute_arr = list(range(len(obs.shape)))
        permute_arr[0], permute_arr[1] = 1, 0
        print(permute_arr)
        obs = np.transpose(obs, permute_arr)
        print(obs.shape)

        new_shape = [obs.shape[0] * obs.shape[1]] + list(obs.shape[2:])
        print(new_shape)
        traj = np.reshape(obs, new_shape)
        self.show_func(trajectory=traj)


class WandbCallback(BaseCallback):
    '''
        call back for logging env reward in SAC Algo
        *** support one env only ***
        excpect for 2 dicts in info:
        - wandb_log:

    '''
    def __init__(self, wandb_proj=None, verbose=0, every_step=False):
        super(WandbCallback, self).__init__(verbose)
        self.wandb_proj = wandb_proj
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_r = 0
        # log evety step / every episode
        self.every_step = every_step

    def _on_step(self) -> bool:
        # print("on_step episode_r: ", self.episode_r)
        # log every step
        logger = {}
        do_log = self.every_step
        self.total_steps += 1
        if 'reward' in self.locals.keys():
            # single env
            self.episode_r += self.locals['reward']
            done = self.locals['done']

        elif 'rewards' in self.locals.keys():
            # support dummy vec env, but need to make sure only size 1
            self.episode_r += self.locals['rewards'][0]
            done = self.locals['dones'][0]

        else:
            assert False, "env locals missing reward / rewards"

        # Log global params that are not relevant to specific env (to be the same as subprocvecenv)
        if 'global_wandb_log' in self.locals['info'].keys():
            for k, v in self.locals['info']['global_wandb_log'].items():
                logger[k] = v

        if 'wandb_log' in self.locals['info'].keys():
            for k, v in self.locals['info']['wandb_log'].items():
                logger['env_0/' + k] = v

        if done or self.every_step:
            logger['env_0/episode_reward'] = self.episode_r  # maybe change to 'learned_episode_reward'
            logger['env_0/total_episodes'] = self.total_episodes
            do_log = True

        if done:
            self.total_episodes += 1
            logger['CallBack/episode_reward'] = self.episode_r  # maybe change to 'learned_episode_reward'
            logger['CallBack/total_episodes'] = self.total_episodes
            logger['CallBack/total_steps'] = self.total_steps
            self.episode_r = 0

        logger['Global_info/global_total_steps'] = self.total_steps

        if do_log and self.wandb_proj is not None:
            wandb.log(logger)

        return True

class WandbVecCallback(BaseCallback):
    '''
        call back for logging env reward in SAC Algo
        supports vec env
        excpect for 2 dicts in every info in infos:
        - wandb_log:

    '''
    def __init__(self, n_envs, wandb_proj=None, verbose=0, every_step=False):
        super(WandbVecCallback, self).__init__(verbose)
        self.n_envs = n_envs
        self.wandb_proj = wandb_proj
        self.total_steps = 0
        self.total_episodes = [0 for _ in range(n_envs)]
        self.episode_r = [0 for _ in range(n_envs)]
        # log every step / every episode
        self.log_every_step = every_step

    def _on_step(self) -> bool:
        # print("on_step episode_r: ", self.episode_r)
        # log every step
        logger = {}
        do_log = self.log_every_step
        self.total_steps += 1
        assert self.n_envs == len(self.locals['rewards']), "n_envs doesn't match trainer local info! " \
                                                            "({} != {})".format(self.n_envs, len(self.locals['rewards']))
        # Log global params that were pushed to env_0 info dict
        if 'global_wandb_log' in self.locals['infos'][0].keys():
            for k, v in self.locals['infos'][0]['global_wandb_log'].items():
                logger[k] = v

        num_of_dones = 0
        for i in range(self.n_envs):
            self.episode_r[i] += self.locals['rewards'][i]

            if self.locals['dones'][i] or self.log_every_step:
                do_log = True
                num_of_dones += 1

                for k, v in self.locals['infos'][i]['wandb_log'].items():
                    logger[f'env_{i}/' + k] = v

                    if do_log:
                        if (f'env_avg/{k}' not in logger.keys()):
                            logger[f'env_avg/{k}'] = v
                        else:
                            logger[f'env_avg/{k}'] += v

                logger[f'env_{i}/episode_reward'] = self.episode_r[i]  # maybe change to 'learned_episode_reward'
                logger[f'env_{i}/total_episodes'] = self.total_episodes[i]
                if (f'env_avg/episode_reward' not in logger.keys()):
                    logger[f'env_avg/episode_reward'] = self.episode_r[i]
                else:
                    logger[f'env_avg/episode_reward'] += self.episode_r[i]

            if self.locals['dones'][i]:
                self.total_episodes[i] += 1
                self.episode_r[i] = 0

        logger['Global_info/global_total_steps'] = self.total_steps

        for k in logger.keys():
            if 'env_avg/' in k and num_of_dones > 0:
                logger[k] = logger[k] / num_of_dones

        if do_log and self.wandb_proj is not None:
            wandb.log(logger)

        return True
