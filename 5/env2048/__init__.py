from gymnasium.envs.registration import register

register(
    id="TwentyFortyEight-v1",
    entry_point="env2048.envs:Env2048",
)
