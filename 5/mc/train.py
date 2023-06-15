from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import gymnasium as gym
import env
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 64)
        self.fc6 = nn.Linear(64, action_size)

    def forward(self, x):
        x = x.view(x.size(0), -1).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  # Experience replay memory
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration vs exploitation factor
        self.epsilon_decay = 0.99  # Epsilon decay rate
        self.epsilon_min = 0.01  # Minimum epsilon value
        self.batch_size = 32  # Batch size for training

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit: choose the best action based on the current Q-values
            state = np.asarray(state, dtype=np.int32)
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = np.asarray(state_batch, dtype=np.int32)
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1)

        q_current = self.model(state_batch).gather(1, action_batch)
        q_next = self.model(next_state_batch).max(1)[0].unsqueeze(1)
        q_target = reward_batch + self.gamma * q_next * (1 - done_batch)

        loss = F.smooth_l1_loss(q_current, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


if __name__ == '__main__':
    num_episodes = 500

    state_size = 16
    action_size = 4

    agent = DQNAgent(state_size, action_size)

    env2048 = gym.make("Gym-v0")

    writer = SummaryWriter()

    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env2048.reset()  # Reset the game and get the initial state
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env2048.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay()

        writer.add_scalar('Reward/Episode', total_reward, episode)

        # print(f"Episode: {episode + 1}, Reward: {total_reward}")

    # After training, you can use the trained agent to play the game
    state = env2048.reset()
    done = False

    while not done:
        action = agent.act(state)
        state, reward, done, info = env2048.step(action)
        env2048.render()
        # Perform any necessary visualization or logging

    print("Game over!")
