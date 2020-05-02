from gym.envs.registration import register

register(
    id='bbo-v0',
    entry_point='gym_bm.envs:BlackBoxOptEnv',
)