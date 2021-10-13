from gym.envs.registration import register

register(
    id='rooms-v0',
    entry_point='src.envs.rooms.rooms:RoomsEnv',
    max_episode_steps=100000,
    kwargs={}
)
register(
    id='random_rooms-v1',
    entry_point='src.envs.random_rooms.random_rooms_env:RandomRoomsEnv',
    reward_threshold=1,
    kwargs={'cmd_args': None}
)
# register(
#     id='hopper-v3',
#     entry_point='src.envs.RP_hopper.RP_hopper:RPHopperEnv',
#     max_episode_steps=1000,
#     reward_threshold=3800.0,
#     kwargs={'Hopper_args': None}
#     #reward_threshold=1,
# )
# register(
#     id='ant_rp-v0',
#     entry_point='src.envs.RP_ant.RP_ant:RPAntEnv',
#     max_episode_steps=1000,
#     reward_threshold=6000.0,
#     kwargs={'Ant_args': None}
# )
# register(
#     id='cheetah_rp-v0',
#     entry_point='src.envs.RP_cheetah.RP_cheetah:RPCheetahEnv',
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
#     kwargs={'Cheetah_args': None}
# )
# register(
#     id='pendulum_rp-v0',
#     entry_point='src.envs.RP_pendulum.RP_pendulum:RPPendulumEnv',
#     max_episode_steps=1000,
#     reward_threshold=950.0,
#     kwargs={'Pendulum_args': None}
# )
# register(
#     id='double_pendulum_rp-v0',
#     entry_point='src.envs.RP_double_pendulum.RP_double_pendulum:RPDoublePendulumEnv',
#     max_episode_steps=1000,
#     reward_threshold=9100.0,
#     kwargs={'Pendulum_args': None}

# )

# register(
#     id='rp_env-v0',
#     entry_point='src.envs.RP_env:get_rp_env',
#     max_episode_steps=1000,
#     reward_threshold=950.0,
#     kwargs={'args': None}
# )