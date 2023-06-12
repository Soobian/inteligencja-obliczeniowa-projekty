from __future__ import annotations

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import gymnasium as gym
import env2048
from torch.utils.tensorboard import SummaryWriter


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(18, 256, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_out = nn.Linear(256, 4)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h)).view(-1, 2048)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        out = self.fc_out(h)
        return out


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(18, 256, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h)).view(-1, 2048)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        out = self.fc_out(h)
        return out


class A2C(nn.Module):
    def __init__(
            self,
            device: torch.device,
            critic_lr: float,
            actor_lr: float,
            n_envs: int,
    ):
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        self.actor = PolicyNetwork().to(device)
        self.critic = ValueNetwork().to(device)

        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(params=self.critic.parameters(), lr=critic_lr)

    def forward(self, x: np.ndarray):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        state_values = self.critic(x)
        action_logits = self.actor(x)
        return state_values, action_logits

    def select_action(self, x: np.ndarray, legal_actions):
        batch_size = len(x)
        state_values, action_logits = self.forward(x)

        selected_actions = torch.zeros(size=(batch_size,), dtype=torch.int32, device=self.device)
        action_logprobs = torch.zeros(size=(batch_size,), dtype=torch.float32, device=self.device)
        entropy = torch.zeros(size=(batch_size,), dtype=torch.float32, device=self.device)

        for i in range(batch_size):
            action_pd = torch.distributions.Categorical(logits=action_logits[i][legal_actions[i]])
            selected_actions[i] = action_pd.sample()
            action_logprobs[i] = action_pd.log_prob(selected_actions[i])
            entropy[i] = action_pd.entropy()

        return selected_actions, action_logprobs, state_values, entropy

    def get_losses(
            self,
            rewards,
            action_log_probs,
            value_preds,
            entropy,
            masks,
            gamma,
            lam,
            ent_coef,
            device,
    ):
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        critic_loss = advantages.pow(2).mean()
        actor_loss = -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()

        return critic_loss, actor_loss

    def update_parameters(self, critic_loss, actor_loss):
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()


if __name__ == '__main__':

    n_envs = 4
    n_updates = 1000
    n_steps_per_update = 128

    gamma = 0.999
    lam = 0.95
    ent_coef = 0.01
    actor_lr = 0.001
    critic_lr = 0.005

    envs = gym.vector.make("TwentyFortyEight-v1", num_envs=n_envs)

    writer = SummaryWriter()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    agent = A2C(device, critic_lr, actor_lr, n_envs)

    envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=n_envs * n_updates)

    critic_losses = []
    actor_losses = []
    entropies = []

    for sample_phase in tqdm(range(n_updates)):
        ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
        masks = torch.zeros(n_steps_per_update, n_envs, device=device)

        if sample_phase == 0:
            states, infos = envs_wrapper.reset(seed=42)

        for step in range(n_steps_per_update):
            actions, action_log_probs, state_value_preds, entropy = agent.select_action(states, infos["legal actions"])

            states, rewards, terminated, truncated, infos = envs_wrapper.step(actions.cpu().numpy())

            ep_value_preds[step] = torch.squeeze(state_value_preds)
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_action_log_probs[step] = action_log_probs
            masks[step] = torch.tensor([not term for term in terminated])

        critic_loss, actor_loss = agent.get_losses(
            ep_rewards,
            ep_action_log_probs,
            ep_value_preds,
            entropy,
            masks,
            gamma,
            lam,
            ent_coef,
            device,
        )
        writer.add_scalar("Loss/Critic", critic_loss.item(), sample_phase)
        writer.add_scalar("Loss/Actor", actor_loss.item(), sample_phase)
        writer.add_scalar("Entropy", entropy.mean().item(), sample_phase)

        agent.update_parameters(critic_loss, actor_loss)

        critic_losses.append(critic_loss.detach().cpu().numpy())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropies.append(entropy.detach().mean().cpu().numpy())
    writer.close()

