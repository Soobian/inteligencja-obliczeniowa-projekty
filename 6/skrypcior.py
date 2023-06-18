import re

import gym
from stable_baselines3 import PPO, DDPG
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from tqdm import tqdm

# Utwórz środowisko
env = gym.make("MountainCarContinuous-v0")

# Definiuj zestawy hiperparametrów
hyperparams = [
    {'learning_rate': 0.01,
     'batch_size':    32,
     "action_noise":  NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]),
                                        sigma=0.1 * np.ones(env.action_space.shape[-1]))},
    {'learning_rate': 0.1,
     'batch_size':    32,
     "action_noise":  NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]),
                                        sigma=0.1 * np.ones(env.action_space.shape[-1]))},
    {'learning_rate': 0.1,
     'batch_size':    128,
     "action_noise":  NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]),
                                        sigma=0.1 * np.ones(env.action_space.shape[-1]))}
]

# Przechowuj wyniki
results = []

# Dla każdego zestawu hiperparametrów
for params in tqdm(hyperparams, desc=f'Hyperparams'):
    rewards = []
    stds = []

    tensorboard_path = re.sub(r'[^\w.-]', '_',
                              f'learning_rate_{params["learning_rate"]}_batch_size_{params["batch_size"]}')
    model = DDPG("MlpPolicy", env, tensorboard_log=tensorboard_path, device='cuda', **params)

    model.learn(total_timesteps=50000, log_interval=1)

    # Powtórz eksperyment 10 razy
    for _ in tqdm(range(15), desc='Experiments'):
        # Testowanie nauczonego modelu przez 10 epizodów
        all_rewards = []
        for _ in tqdm(range(10), desc='Testing'):
            obs = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            all_rewards.append(episode_reward)

            # Dodaj wyniki do listy
        rewards.append(np.mean(all_rewards))
        stds.append(np.std(all_rewards))

        # Dodaj wyniki do głównej listy
    results.append((rewards, stds))

# Narysuj krzywe uczenia po wszystkich eksperymentach
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