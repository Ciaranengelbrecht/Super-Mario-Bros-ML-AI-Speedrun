import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
import warnings

# Suppress numpy deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def make_action(obs, info, step, env, prev_action):
    JUMP_COOLDOWN = 20
    last_jump_step = -JUMP_COOLDOWN
    SAFE_LANDING_SPACE = 40
    RUNNING_START_DISTANCE = 40
    recent_actions = []

    mario_pos = info.get("mario_pos")
    if not mario_pos:
        if step % 10 == 0:
            return env.action_space.sample()
        else:
            return prev_action

    mario_x, mario_y = mario_pos

    enemies = info.get("enemies", [])
    pipes = info.get("pipes", [])
    blocks = info.get("blocks", [])
    q_blocks = [block for block in blocks if block["type"] == "Q"]

    overhead_blocks = [block for block in blocks if mario_x < block["x"] < mario_x + 40 and abs(block["y"] - mario_y) < 10]
    q_blocks_overhead = [block for block in q_blocks if mario_x < block["x"] < mario_x + 20 and block["y"] - mario_y < 40]

    if q_blocks_overhead and step - last_jump_step > JUMP_COOLDOWN:
        last_jump_step = step
        return 4

    for enemy in enemies:
        distance_x = enemy["x"] - mario_x
        distance_y = abs(enemy["y"] - mario_y)

        # Check for enemies above on blocks
        if enemy["y"] < mario_y and distance_x < 50:
            return 6  # Move left to avoid enemies above falling down

        if prev_action == 4 and 0 < distance_x < SAFE_LANDING_SPACE and distance_y < 15:
            return 6  # Move left to ensure safe landing

        if 30 < distance_x < 100 and distance_y < 20 and step - last_jump_step > JUMP_COOLDOWN:
            last_jump_step = step
            return 4  # Predictive jump

    for pipe in pipes:
        distance_x = pipe["x"] - mario_x
        if 0 < distance_x < 40:
            if pipe["height"] > 2 and step - last_jump_step > JUMP_COOLDOWN:
                last_jump_step = step
                return 4

    tiles = info.get("tiles", [])
    tiles_ahead = tiles[mario_y : mario_y + 2, mario_x + 1 : mario_x + 100]
    gap_size = sum(row.count(0) for row in tiles_ahead)

    if gap_size > 10:
        if gap_size < RUNNING_START_DISTANCE:
            return 6  # Move left for running start
        elif step - last_jump_step > JUMP_COOLDOWN:
            last_jump_step = step
            return 4

    if prev_action == 4 and gap_size > 4:
        return 5

    if len(recent_actions) > 10:
        recent_actions.pop(0)
    recent_actions.append(prev_action)

    if recent_actions.count(recent_actions[-1]) > 5:
        return env.action_space.sample()

    # Early game enemy strategy
    if mario_x < 100:
        if overhead_blocks and distance_y < 15 and 0 < distance_x < 60:
            return 4  # Jump early to avoid first enemy

    return 1


################################################################################

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs = None
done = True
env.reset()
action = env.action_space.sample()  # Initialize a random action
for step in range(100000):
    if obs is not None:
        action = make_action(obs, info, step, env, action)
    else:
        action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if info.get('flag_get'):
        print("Mario has reached the flag!")


    #time.sleep(0.001)

    done = terminated or truncated
    if done:
        env.reset()
