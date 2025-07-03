import time
import gym
import numpy as np
from gym.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from utilities import *

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

if __name__ == '__main__':
    start = time.perf_counter()

    # Configuration
    env_id = "CartPole-v1"
    seed = 0
    set_random_seed(seed=seed)
    model_name = 'PPO'

    # Logging directory
    log_dir = 'logs/' + model_name + '_' + str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())) + '/'

    # Create vectorized training environment (no render mode needed here)
    env = make_vec_env(env_id, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv)

    # Agent
    model_hyperparameters = {'policy': 'MlpPolicy'}
    model = PPO(**model_hyperparameters, env=env, verbose=0, tensorboard_log=log_dir)
    model.name = model_name

    # Logger
    logger = configure(log_dir, ["csv", "tensorboard"])
    log_steps_callback = LogStepsCallback(log_dir=log_dir)
    tqdm_callback = TqdmCallback()
    model.set_logger(logger)
    save_dict_to_file(model_hyperparameters, path=log_dir)  # log hyperparameters

    # Learn
    print("Τraining starts!")
    model.learn(total_timesteps=500_000, callback=[tqdm_callback, log_steps_callback])
    print("Τraining ends!")

    # Plot learning curve
    learning_curve(log_dir=log_dir)

    # Render episodes and save video
    render_episodes = True
    num_steps = 500
    if render_episodes:
        # Create a separate environment for recording (must not be VecEnv)
        render_env = gym.make("CartPole-v1", render_mode="rgb_array")
        render_env = RecordVideo(
            render_env,
            video_folder=log_dir + "/video",
            episode_trigger=lambda episode_id: True,
            name_prefix="cartpole"
        )

        obs, info = render_env.reset()
        for _ in range(num_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = render_env.step(int(action))
            done = terminated or truncated
            if done:
                obs, info = render_env.reset()
        render_env.close()

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
