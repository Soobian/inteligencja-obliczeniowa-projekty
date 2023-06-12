import gymnasium as gym
import env2048

# game with user input

if __name__ == '__main__':
    env = gym.make("TwentyFortyEight-v1")
    env.reset()
    env.render()
    done = False
    while not done:
        key = input()
        if key == "w":
            action = 0
        elif key == "s":
            action = 1
        elif key == "d":
            action = 2
        elif key == "a":
            action = 3
        else:
            print("Wrong key")
            continue
        obs, reward, done, info, _ = env.step(action)
        env.render()
        print("Reward: ", reward)

