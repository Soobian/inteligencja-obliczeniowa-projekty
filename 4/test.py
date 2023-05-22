from pettingzoo.atari import pong_v3
import numpy as np



env = pong_v3.env(num_players=2)

obs = env.reset()
done = False

while not done:
    action_1 = 0
    action_2 = 0
    if obs[0][0] is not None:
        action_1 = np.random.choice(env.action_spaces[0])
    if obs[1][0] is not None:
        action_2 = np.random.choice(env.action_spaces[1])

    obs, reward, done, info = env.step([action_1, action_2])

    env.render()

env.close()
