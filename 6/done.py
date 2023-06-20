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

results_rewards = []
results_timesteps = []
results_fps = []

for params in tqdm(hyperparams, desc=f'Hyperparams'):
    all_rewards = []
    all_stds = []
    timesteps = []
    fpss = []
    tensorboard_path = re.sub(r'[^\w.-]', '_',
                              f'learning_rate_{params["learning_rate"]}_batch_size_{params["batch_size"]}')

    for _ in tqdm(range(10), desc='Experiments'):
        model = DDPG("MlpPolicy", env, tensorboard_log=tensorboard_path, device='cuda', **params)
        logger = configure("results/", ["csv"])
        model.set_logger(logger)

        model.learn(total_timesteps=50000, log_interval=1)

        output = pd.read_csv("results/progress.csv", sep=',')
        ep_rew = output['rollout/ep_rew_mean'].to_numpy()
        timestep = output['rollout/ep_len_mean'].to_numpy()
        fps = output['time/fps'].to_numpy()
        all_rewards.append(ep_rew)
        timesteps.append(timestep)

        fpss.append(fps)

    title = f'learning rate: {params["learning_rate"]}, batch size: {params["batch_size"]}, ' \
            f'\nnoise: {params["action_noise"].__class__.__name__ if params["action_noise"] is not None else "None"}'
    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    plt.plot(mean_rewards, label='Reward')
    plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

    mean_timesteps = np.mean(timesteps, axis=0)
    std_timesteps = np.std(timesteps, axis=0)
    plt.plot(mean_timesteps, label='Timesteps')
    plt.fill_between(range(len(mean_timesteps)), mean_timesteps - std_timesteps, mean_timesteps + std_timesteps,
                     alpha=0.3)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Timesteps')
    plt.legend()
    plt.show()

    mean_fps = np.mean(fpss, axis=0)
    std_fps = np.std(fpss, axis=0)

    plt.plot(mean_fps, label='FPS')
    plt.fill_between(range(len(mean_fps)), mean_fps - std_fps, mean_fps + std_fps, alpha=0.3)
    plt.title("FPS" + title)
    plt.xlabel('Episode')
    plt.ylabel('FPS')
    plt.legend()
    plt.show()

    results_rewards.append((mean_rewards, std_rewards))
    results_timesteps.append((mean_timesteps, std_timesteps))
    results_fps.append((mean_fps, std_fps))


def plot_results(hyperparams, results, metric_name):
    fig, axs = plt.subplots(figsize=(8, 6))
    for i, (params, (mean_metric, std_metric)) in enumerate(zip(hyperparams, results)):
        title = f'learning rate: {params["learning_rate"]}, batch size: {params["batch_size"]}, ' \
                f'\nnoise: {params["action_noise"].__class__.__name__ if params["action_noise"] is not None else "None"}'
        axs.plot(mean_metric, label=metric_name)
        axs.fill_between(range(len(mean_metric)), mean_metric - std_metric, mean_metric + std_metric, alpha=0.3)
        axs.set_title(title)
        axs.set_xlabel('Step/Episode')
        axs.set_ylabel(metric_name)
        axs.legend()
    plt.tight_layout()
    plt.show()


plot_results(hyperparams, results_rewards, 'Reward')
plot_results(hyperparams, results_timesteps, 'Timesteps')
plot_results(hyperparams, results_fps, 'FPS')
