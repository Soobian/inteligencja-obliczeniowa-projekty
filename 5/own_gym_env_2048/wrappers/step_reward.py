import gymnasium as gym


class StepReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self):
        return 1
