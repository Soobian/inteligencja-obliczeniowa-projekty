import os
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from pettingzoo.atari import pong_v3


def _get_agents(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    if agent_learn is None:
        # model
        net = Net(
            state_shape=observation_space.shape or observation_space.n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=[128, 128, 256,128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.7,
            
            estimation_step=3,
            target_update_freq=320,
        )

    if agent_opponent is None:
        agent_opponent = RandomPolicy()

    agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(pong_v3.environment())


if __name__ == "__main__":
    # ======== Step 1: Environment setup =========
    print(f'Environment setup')
    print(f'Using device "{torch.device("cuda" if torch.cuda.is_available() else "cpu")}"')
    train_envs = DummyVectorEnv([_get_env for _ in range(5)])
    test_envs = DummyVectorEnv([_get_env for _ in range(1)])

    # seed
    seed = 420
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()
    print(f'Agent setup')

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(100, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    result = train_collector.collect(n_step=64 * 10)  # batch size * training_num
    rews, lens = result["rews"], result["lens"]

    print(f'Collector setup')
    log_path = os.path.join("log", "pong", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("agent", str(policy))
    logger=TensorboardLogger(writer)

    # ======== Step 4: Callback functions setup =========
    print(f'Callback functions setup')

    def save_best_fn(policy):
        model_save_path = os.path.join("log", "rps", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "rps", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 100

    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 1]

    # ======== Step 5: Run the trainer =========
    print(f'Run the trainer...')

    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=700,
        step_per_collect=10,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
        logger=logger,
        
    )

    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")