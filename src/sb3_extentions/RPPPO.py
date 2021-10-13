from stable_baselines3 import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.utils import safe_mean

class RPPPO(PPO):

#TODO: NEED TO ADD T RANDOM ROOMS TO EPISODE INFO THE SUM OF SPECIAL R 
    def _dump_logs(self):
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            logger.record("rollout/ep_special_r_mean", safe_mean([ep_info["special_r"] for ep_info in self.ep_info_buffer]))
        super(SAC,self)._dump_logs()