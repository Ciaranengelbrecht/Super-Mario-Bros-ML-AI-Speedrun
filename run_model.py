
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
import time
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.wrappers import ResizeObservation
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Step 1: Environment Wrappers, adjust based on what we trained with
def make_mario_env():
    env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="none")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    return env

env = DummyVecEnv([make_mario_env]) 
#env = VecFrameStack(env, 4, channels_order='last')

# Load the trained model
model = PPO.load('./train/best_model_20000000.zip')

state = env.reset()
start_time = time.time()
action_counter = 0
cumulative_reward = 0
episode_number = 0 
current_kills = 0
previous_score = 0

actions_taken = []
in_game_times = []
average_rewards = []
distances_reached = []
num_coins_collected = []
kills_per_episode = []

while True: 
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    action_counter += 1
    cumulative_reward += reward[0] 
    score_difference = info[0]['score'] - previous_score
    if score_difference == 100:
        current_kills += 1
    previous_score = info[0]['score']

    if "flag_get" in info[0] and info[0]["flag_get"]:
            print("Mario reached the flag!")
            time_taken_game = 400 - info[0]['time']
            in_game_times.append(time_taken_game)
            print(f"Time taken in game: {time_taken_game}")
    if done:
        episode_number += 1
        print(f"Episode {episode_number}")
        distance_reached = info[0]['x_pos']
        #print(f"Distance reached: {distance_reached}")
        distances_reached.append(distance_reached)    
        elapsed_time = time.time() - start_time
        #print(f"Real-world time taken since initiating the run: {elapsed_time:.2f} seconds")
        current_score = info[0]['score']
        #print(f"Current Score: {current_score}")
        #print(f"Total actions taken: {action_counter}")
        average_reward = cumulative_reward# / action_counter if action_counter != 0 else 0 #use this for avg reward per action
        #print(f"Average reward: {average_reward}")
        coins_collected = info[0]['coins']

        # Append metrics to lists
        actions_taken.append(action_counter)
        average_rewards.append(average_reward)
        num_coins_collected.append(coins_collected)
        kills_per_episode.append(current_kills)

        # Reset the action counter and cumulative reward after each episode
        action_counter = 0
        cumulative_reward = 0
        current_kills = 0 
        previous_score = 0 
        state = env.reset()

        # break the loop after specified number of episodes to plot the graphs
        if len(actions_taken) > 500:  
            break
env.close()

# Plotting the metrics
plt.figure(figsize=(14, 8))
#episode_ticks = range(1, len(actions_taken) + 1) 

plt.subplot(2, 3, 1)
plt.plot(actions_taken, label='Actions Taken')
plt.xlabel('Episode')
plt.ylabel('Actions Taken')
plt.title('Actions Taken per Episode')
#plt.xticks(episode_ticks)

plt.subplot(2, 3, 2)
plt.plot(in_game_times, label='In-Game Time', color='orange')
plt.xlabel('Episode of Reaching Flag')
plt.ylabel('Time (in-game time)')
plt.title('In-Game Time to Reach Flag')

plt.subplot(2, 3, 3)
plt.plot(average_rewards, label='Episode Reward', color='green')
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title(' Reward per Episode')
#plt.xticks(episode_ticks)

plt.subplot(2, 3, 4)
plt.plot(distances_reached, label='Distance Reached', color='purple')
plt.xlabel('Episode')
plt.ylabel('Distance Reached')
plt.title('Distance Reached per Episode')
#plt.xticks(episode_ticks)

plt.subplot(2, 3, 5)
plt.plot(num_coins_collected, label='Coins Collected', color='gold')
plt.xlabel('Episode')
plt.ylabel('Coins Collected')
plt.title('Coins Collected per Episode')

plt.subplot(2, 3, 6)
plt.plot(kills_per_episode, label='Kills', color='red')
plt.xlabel('Episode')
plt.ylabel('Kills')
plt.title('Kills per Episode')

plt.tight_layout()
plt.show()
