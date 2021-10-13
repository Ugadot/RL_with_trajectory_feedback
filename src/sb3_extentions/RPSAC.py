from stable_baselines3 import SAC
from stable_baselines3.common import logger
from stable_baselines3.common.utils import safe_mean
import numpy as  np
class RPSAC(SAC):

    def _dump_logs(self):
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            logger.record("rollout/ep_sparse_rew_mean", safe_mean([ep_info["sparse_r"] for ep_info in self.ep_info_buffer]))
            logger.record("rollout/ep_max_distance", np.max(np.array([ep_info["distance"] for ep_info in self.ep_info_buffer])))
        
        super(RPSAC,self)._dump_logs()