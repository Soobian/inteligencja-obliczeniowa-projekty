import re
import gym
from stable_baselines3 import DDPG
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from tqdm import tqdm
import pandas as pd

# Utwórz środowisko
env = gym.make("MountainCarContinuous-v0")

# Definiuj zestawy hiperparametrów
hyperparams = [
    {'learning_rate': 0.001,
     'batch_size':    128,
     "action_noise":  OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[-1]),
                                                   sigma=0.1 * np.ones(env.action_space.shape[-1]))},
    {'learning_rate': 0.001,
     'batch_size':    128,
     "action_noise":  NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]),
                                        sigma=0.1 * np.ones(env.action_space.shape[-1]))},
    {'learning_rate': 0.001,
     'batch_size':    128,
     "action_noise":  None}
]

results = []

for params in tqdm(hyperparams, desc=f'Hyperparams'):
    all_rewards =[]
    all_stds = []
    tensorboard_path = re.sub(r'[^\w.-]', '_',
                              f'learning_rate_{params["learning_rate"]}_batch_size_{params["batch_size"]}')
    model = DDPG("MlpPolicy", env, tensorboard_log=tensorboard_path, device='cpu', **params)
    logger = configure("results/", ["csv"])
    model.set_logger(logger)

    for _ in tqdm(range(2), desc='Experiments'):

        model.learn(total_timesteps=999, log_interval=1)

        output = pd.read_csv("results/progress.csv", sep=',')
        ep_rew = output['rollout/ep_rew_mean'].to_numpy()
        all_rewards.append(ep_rew)

    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    plt.plot(mean_rewards, label='Reward')
    plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
    plt.title("Learning rate: {}, batch size: {}".format(params["learning_rate"], params["batch_size"]))
    plt.xlabel('Step/Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

    results.append((mean_rewards, std_rewards))

    # Print episode time and time for each time step
    print(f'Hyperparams: learning rate: {params["learning_rate"]}, batch size: {params["batch_size"]}')
    for i, ep_rew in enumerate(all_rewards):
        print(f'Episode {i + 1} - Time: {model.ep_info_buffer[i]["episode"]["l"]}s')
        for j, timestep_rew in enumerate(ep_rew):
            print(f'Time Step {j + 1} - Time: {model.ep_info_buffer[i]["t"][j]}s')


fig, axs = plt.subplots(len(hyperparams), figsize=(8, 6))
for i, (params, (mean_rewards, std_rewards)) in enumerate(zip(hyperparams, results)):
    axs[i].plot(mean_rewards, label='Reward')
    axs[i].fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
    axs[i].set_title(
        f'Hyperparams: learning rate: {params["learning_rate"]}, batch size: {params["batch_size"]}')
    axs[i].set_xlabel('Step/Episode')
    axs[i].set_ylabel('Reward')
    axs[i].legend()
plt.tight_layout()
plt.show()
