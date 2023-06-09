from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# Create an instance of your custom environment
env = gym.make("TwentyFortyEight-v0")

# Define the RL algorithm and configure hyperparameters
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("custom_env_model")

# Load the saved model
loaded_model = PPO.load("custom_env_model")

# Evaluate the trained model
mean_reward, _ = evaluate_policy(loaded_model, env, n_eval_episodes=10)

print("Mean reward:", mean_reward)
