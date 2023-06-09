import gymnasium as gym
import numpy as np


class ConvObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.MultiBinary([18, 4, 4])

    def observation(self, obs):
        conv_obs = np.zeros(shape=(18, 4, 4), dtype=np.int32)
        for x in range(4):
            for y in range(4):
                conv_obs[obs[x][y]][x][y] = 1.0
        return conv_obs
