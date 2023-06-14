import gymnasium as gym

gym.envs.register(
     id='Gym-v0',
     entry_point='env.env2048:Gym2048',
)