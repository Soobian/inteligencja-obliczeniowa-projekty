from gymnasium.envs.registration import register

register(
    id="TwentyFortyEight-v0",
    entry_point="own_gym_env_2048.envs:TwentyFortyEight",
)
