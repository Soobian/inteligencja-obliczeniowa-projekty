import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
from time import sleep
from matplotlib import animation


def run_animation(experience_buffer):
    time_lag = .05
    for experience in experience_buffer:
        clear_output(True)
        plt.imshow(experience['frame'])
        plt.axis('off')
        plt.show()

        print(f"Episode: {experience['episode']}/{experience_buffer[-1]['episode']}")
        print(f"Epoch: {experience['epoch']}/{experience_buffer[-1]['epoch']}")
        print(f"State: {experience['state']}")
        print(f"Action: {experience['action']}")
        print(f"Reward: {experience['reward']}")

        sleep(time_lag)


def store_episode_as_gif(experience_buffer, path='./', filename='animation.gif'):
    fps = 5
    dpi = 30
    interval = 50

    frames = []
    for experience in experience_buffer:
        frames.append(experience['frame'])

    plt.figure(figsize=(frames[0].shape[1], frames[0].shape[0]), dpi=dpi)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=interval)
    anim.save(path + filename, writer='imagemagick', fps=fps)


env = gym.make('Taxi-v3', render_mode="rgb_array")
state, _ = env.reset()
print("State space: {}".format(env.observation_space))
print("Action space: {}".format(env.action_space))

action = env.action_space.sample()
next_state, reward, done, _, _ = env.step(action)

# Print output
print("State: {}".format(state))
print("Action: {}".format(action))
print("Reward: {}".format(reward))

frame = env.render()
plt.imshow(frame)
plt.axis('off')
plt.show()

epoch = 0
num_failed_dropoffs = 0
experience_buffer = []
cum_rewards = 0

done = False

state, _ = env.reset()

while not done:
    action = env.action_space.sample()

    state, reward, done, _, _ = env.step(action)
    cum_rewards += reward

    experience_buffer.append({'frame': env.render(),
                              'episode': 1,
                              'epoch': epoch,
                              'state': state,
                              'action': action,
                              'reward': cum_rewards
                              })
    if reward == -10:
        num_failed_dropoffs += 1
    epoch += 1

# store_episode_as_gif(experience_buffer, './out/')
run_animation(experience_buffer)
print('# epochs : {}'.format(epoch))
print('# failed dropoffs: {}'.format(num_failed_dropoffs))



"""Training the agent"""
q_table = np.zeros([env.observation_space.n, env.action_space.n])
experience_buffer = []

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 1.0  # Discount rate
epsilon = 0.1  # Exploration rate
num_episodes = 10000  # Number of episodes

# Output for plots
cum_rewards = np.zeros([num_episodes])
total_epochs = np.zeros([num_episodes])

for episode in range(1, num_episodes + 1):
    # Reset environment
    state, info = env.reset()
    epoch = 0
    num_failed_dropoffs = 0
    done = False
    cum_reward = 0

    while not done:

        if random.uniform(0, 1) < epsilon:
            "Basic exploration [~0.47m]"
            action = env.action_space.sample()  # Sample random action (exploration)

            "Exploration with action mask [~1.52m]"
        # action = env.action_space.sample(env.action_mask(state)) "Exploration with action mask"
        else:
            "Exploitation with action mask [~1m52s]"
            # action_mask = np.where(info["action_mask"]==1,0,1) # invert
            # masked_q_values = np.ma.array(q_table[state], mask=action_mask, dtype=np.float32)
            # action = np.ma.argmax(masked_q_values, axis=0)

            "Exploitation with random tie breaker [~1m19s]"
            #  action = np.random.choice(np.flatnonzero(q_table[state] == q_table[state].max()))

            "Basic exploitation [~47s]"
            action = np.argmax(q_table[state])  # Select best known action (exploitation)

        next_state, reward, done, _, info = env.step(action)

        cum_reward += reward

        old_q_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)

        q_table[state, action] = new_q_value

        if reward == -10:
            num_failed_dropoffs += 1

        state = next_state
        epoch += 1

        total_epochs[episode - 1] = epoch
        cum_rewards[episode - 1] = cum_reward

    if episode % 100 == 0:
        clear_output(wait=True)
        print(f"Episode #: {episode}")

print("\n")
print("===Training completed.===\n")
store_episode_as_gif(experience_buffer)

# Plot reward convergence
plt.title("Cumulative reward per episode")
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.plot(cum_rewards)
plt.show()

# Plot epoch convergence
plt.title("# epochs per episode")
plt.xlabel("Episode")
plt.ylabel("# epochs")
plt.plot(total_epochs)
plt.show()