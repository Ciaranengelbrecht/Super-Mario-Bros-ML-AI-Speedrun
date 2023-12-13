import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
import warnings
import matplotlib.pyplot as plt

# Suppress numpy deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def make_action(obs, info, step, env, prev_action):

    RUNNING_START_DISTANCE = 50

    mario_pos = info.get("mario_pos")
    if not mario_pos:
        return env.action_space.sample() if step % 10 == 0 else prev_action

    mario_x, mario_y = mario_pos

    enemies = info.get("enemies", [])
    pipes = info.get("pipes", [])
    blocks = info.get("blocks", [])
    q_blocks = [block for block in blocks if block["type"] == "Q"]

    # Jumping logic for power-ups
    for block in q_blocks:
        distance_x = block["x"] - mario_x
        distance_y = block["y"] - mario_y
        if 0 < distance_x < 20 and 0 < distance_y < 40:
            return 4

    # Enemy avoidance
    for enemy in enemies:
        distance_x = enemy["x"] - mario_x
        distance_y = abs(enemy["y"] - mario_y)

        if 0 < distance_x < 30 and distance_y < 20:
            return 4

    # Gap crossing
    tiles = info.get("tiles", [])
    tiles_ahead = tiles[mario_y : mario_y + 2, mario_x + 1 : mario_x + 100]
    gap_size = sum(row.count(0) for row in tiles_ahead)

    if gap_size > 10 and mario_x > RUNNING_START_DISTANCE:
        return 4

    # Move right by default
    return 1

################################################################################
actions_per_episode = []
scores_per_episode = []
distances_per_episode = []
coins_per_episode = []
in_game_times = []
kills_per_episode = []

env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="none")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs = None
done = True
env.reset()
action = env.action_space.sample()

action_count = 0
episode_count = 0
max_episodes = 1000 
current_kills = 0
previous_score = 0

for step in range(1000000):
    if obs is not None:
        action = make_action(obs, info, step, env, action)
    else:
        action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    action_count += 1
    score_difference = info['score'] - previous_score
    if score_difference == 100:
        current_kills += 1

    previous_score = info['score']
    if info.get('flag_get'):
            print("Mario reached the flag!")
            time_taken_game = 400 - info['time']
            in_game_times.append(time_taken_game)
            print(f"Time taken in game: {time_taken_game}")

    done = terminated or truncated
    if done:
        episode_count += 1
        print(f"Episode {episode_count} finished after {action_count} actions")

        # Record metrics
        actions_per_episode.append(action_count)
        scores_per_episode.append(info['score'])
        distances_per_episode.append(info['x_pos'])
        coins_per_episode.append(info['coins'])
        kills_per_episode.append(current_kills)
        

        action_count = 0
        current_kills = 0 
        previous_score = 0 
        obs = env.reset()

        if episode_count >= max_episodes:
            break

env.close()

# Plotting the metrics
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.plot(actions_per_episode, label='Actions Taken')
plt.xlabel('Episode')
plt.ylabel('Actions Taken')
plt.title('Actions Taken per Episode')

plt.subplot(2, 3, 2)
plt.plot(in_game_times, label='In-Game Time', color='orange')
plt.xlabel('Episode of Reaching Flag')
plt.ylabel('Time (in-game time)')
plt.title('In-Game Time to Reach Flag')


plt.subplot(2, 3, 3)
plt.plot(scores_per_episode, label='Episode Reward', color='green')
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title(' Reward per Episode')

plt.subplot(2, 3, 4)
plt.plot(distances_per_episode, label='Distance Reached', color='purple')
plt.xlabel('Episode')
plt.ylabel('Distance Reached')
plt.title('Distance Reached per Episode')

plt.subplot(2, 3, 5)
plt.plot(coins_per_episode, label='Coins Collected', color='gold')
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