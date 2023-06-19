import re

import gym
from stable_baselines3 import DDPG
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from tqdm import tqdm
import pandas as pd

# Utwórz środowisko
env = gym.make("MountainCarContinuous-v0")

# Definiuj zestawy hiperparametrów
hyperparams = [
    {'learning_rate': 0.001,
     'batch_size':    32,
     "action_noise":  NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]),
                                        sigma=0.1 * np.ones(env.action_space.shape[-1]))},
    {'learning_rate': 0.01,
     'batch_size':    32,
     "action_noise":  NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]),
                                        sigma=0.1 * np.ones(env.action_space.shape[-1]))},
    {'learning_rate': 0.001,
     'batch_size':    128,
     "action_noise":  NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]),
                                        sigma=0.1 * np.ones(env.action_space.shape[-1]))}
]

results = []

for params in tqdm(hyperparams, desc=f'Hyperparams'):
    rewards = []
    stds = []

    for _ in tqdm(range(5), desc='Experiments'):
        tensorboard_path = re.sub(r'[^\w.-]', '_',
                                  f'learning_rate_{params["learning_rate"]}_batch_size_{params["batch_size"]}')
        model = DDPG("MlpPolicy", env, tensorboard_log=tensorboard_path, device='cuda', **params)
        logger = configure("results/", ["csv"])
        model.set_logger(logger)

        model.learn(total_timesteps=50000, log_interval=1)

        output = pd.read_csv("results/progress.csv", sep=',')
        ep_rew = output['rollout/ep_rew_mean'].to_numpy()
        rewards.append(np.mean(ep_rew))
        stds.append(np.std(ep_rew))

    results.append((rewards, stds))

fig, axs = plt.subplots(len(hyperparams), figsize=(8, 6))
for i, (params, (rewards, stds)) in enumerate(zip(hyperparams, results)):
    axs[i].plot(rewards, label='Reward')
    axs[i].fill_between(range(len(rewards)), np.array(rewards) - np.array(stds), np.array(rewards) + np.array(stds),
                        alpha=0.3)
    axs[i].set_title(f'Hyperparams: learning rate: {params["learning_rate"]}, batch size: {params["batch_size"]}')
    axs[i].set_xlabel('Step/Episode')
    axs[i].set_ylabel('Reward')
    axs[i].legend()
plt.tight_layout()
plt.show()