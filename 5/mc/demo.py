import gymnasium as gym
import env

# game with user input

if __name__ == '__main__':
    env = gym.make("Gym-v0", render_mode="rgb_array")
    env.reset()
    env.render()
    done = False
    while not done:
        key = input()
        if key == "w":
            action = 0
        elif key == "s":
            action = 1
        elif key == "a":
            action = 2
        elif key == "d":
            action = 3
        else:
            print("Wrong key")
            continue
        obs, reward, done, info = env.step(action)
        env.render()
        print("Reward: ", reward)

#         if action == 0:  # Move up
#             self._move_up()
#         elif action == 1:  # Move down
#             self._move_down()
#         elif action == 2:  # Move left
#             self._move_left()
#         elif action == 3:  # Move right
#             self._move_right()