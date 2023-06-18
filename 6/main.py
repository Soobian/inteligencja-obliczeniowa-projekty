import gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make("MountainCarContinuous-v0")

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG(
    "MlpPolicy",
    env=env,
    learning_rate=0.001,
    gamma=0.99,
    batch_size=64,
    action_noise=action_noise,
    verbose=1)
model.learn(total_timesteps=50000, log_interval=1)
model.save("ddpg_pendulum")
vec_env = model.get_env()

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    env.render()
