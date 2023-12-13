
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy
import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack
from gym.wrappers import TimeLimit
from gym.wrappers import GrayScaleObservation
from tqdm import tqdm
from gym.wrappers import ResizeObservation
from stable_baselines3 import DQN



class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info



class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):  
        super(ProgressBarCallback, self).__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        super(ProgressBarCallback, self)._on_training_start()
        self.pbar = tqdm(total=self.total_timesteps)

    def _on_step(self):
        if self.pbar.n < self.total_timesteps:
            self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()



class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True
    
CHECKPOINT_DIR = './train/0001v0'

# Environment Wrapper
def make_mario_env():
    env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="none")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    #env = TimeLimit(env, max_episode_steps=2000) # use this to limit episode length
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    return env

env = DummyVecEnv([make_mario_env])  
env = SkipFrame(env, skip=4)
env = VecMonitor(env)
env = VecFrameStack(env, 4, channels_order='last')

LEARNING_RATE = 0.0001
GAE = 1.0
ENT_COEF = 0.01
N_STEPS = 512
GAMMA = 0.9
BATCH_SIZE = 64
N_EPOCHS = 10
# Use this to check if a specific checkpoint exists to continue training
#previous_checkpoint = os.path.join(CHECKPOINT_DIR, 'best_model_300000')
#if os.path.exists(previous_checkpoint + ".zip"):
#    print(f"Loading model from: {previous_checkpoint}")
#    model = PPO.load(previous_checkpoint, env)
#else:
model = PPO(CnnPolicy, env, verbose=2, tensorboard_log="./tensorboard/", learning_rate=LEARNING_RATE, n_steps=N_STEPS,
              batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE, ent_coef=ENT_COEF)
# Use this for DQN model
# model = DQN('CnnPolicy', env, verbose=2, tensorboard_log="./tensorboard/", learning_rate=LEARNING_RATE, buffer_size=100000, gamma=0.99, batch_size=64)

train_callback = TrainAndLoggingCallback(check_freq=500000, save_path=CHECKPOINT_DIR)
total_timesteps=20000000
#eval callback to save model on every next best model (slows down training)
#eval_env = DummyVecEnv([make_mario_env])
#eval_env = VecMonitor(eval_env)
#eval_callback = EvalCallback(eval_env, best_model_save_path=CHECKPOINT_DIR, log_path=CHECKPOINT_DIR, eval_freq=2000, n_eval_episodes=5, deterministic=True)
progress_callback = ProgressBarCallback(total_timesteps)

# Combining callbacks
callbacks = [train_callback, progress_callback]
model.learn(total_timesteps, callback=callbacks, tb_log_name="0001v0")
