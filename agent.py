import os
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback


class PPOTrainer:

    def __init__(self, train_env, eval_env=None, log_dir="./logs"):

        self.train_env = train_env
        self.eval_env  = eval_env
        self.log_dir   = log_dir
        self.model     = None

        os.makedirs(log_dir, exist_ok=True)


    def build(self):

        self.model = PPO(
            policy        = "MlpPolicy",
            env           = self.train_env,
            learning_rate = 3e-4,
            n_steps       = 2048,
            batch_size    = 64,
            n_epochs      = 10,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.20,
            ent_coef      = 0.01,
            vf_coef       = 0.50,
            max_grad_norm = 0.50,
            policy_kwargs = {"net_arch": [256, 256, 128]},
            verbose       = 1,
        )

        return self.model


    def train(self, total_timesteps=200_000, save=True):

        if self.model is None:
            self.build()

        callbacks = []

        if self.eval_env is not None:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path = self.log_dir,
                log_path             = self.log_dir,
                eval_freq            = 5000,
                n_eval_episodes      = 3,
                deterministic        = True,
                verbose              = 0,
            )
            callbacks.append(eval_callback)

        print(f"Training PPO for {total_timesteps:,} timesteps ...")

        self.model.learn(
            total_timesteps = total_timesteps,
            callback        = callbacks or None,
            progress_bar    = True,
        )

        if save:
            save_path = os.path.join(self.log_dir, "ppo_portfolio")
            self.model.save(save_path)
            print(f"Model saved to {save_path}.zip")

        return self.model


    def load(self, path):

        self.model = PPO.load(path, env=self.train_env)
        print(f"Model loaded from {path}")

        return self.model


class A2CTrainer:

    def __init__(self, train_env, eval_env=None, log_dir="./logs"):

        self.train_env = train_env
        self.eval_env  = eval_env
        self.log_dir   = log_dir
        self.model     = None

        os.makedirs(log_dir, exist_ok=True)


    def build(self):

        self.model = A2C(
            policy        = "MlpPolicy",
            env           = self.train_env,
            learning_rate = 7e-4,
            n_steps       = 5,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            ent_coef      = 0.01,
            vf_coef       = 0.50,
            max_grad_norm = 0.50,
            policy_kwargs = {"net_arch": [256, 256, 128]},
            verbose       = 1,
        )

        return self.model


    def train(self, total_timesteps=200_000, save=True):

        if self.model is None:
            self.build()

        callbacks = []

        if self.eval_env is not None:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path = self.log_dir,
                log_path             = self.log_dir,
                eval_freq            = 5000,
                n_eval_episodes      = 3,
                deterministic        = True,
                verbose              = 0,
            )
            callbacks.append(eval_callback)

        print(f"Training A2C for {total_timesteps:,} timesteps ...")

        self.model.learn(
            total_timesteps = total_timesteps,
            callback        = callbacks or None,
            progress_bar    = True,
        )

        if save:
            save_path = os.path.join(self.log_dir, "a2c_portfolio")
            self.model.save(save_path)
            print(f"Model saved to {save_path}.zip")

        return self.model


    def load(self, path):

        self.model = A2C.load(path, env=self.train_env)
        print(f"Model loaded from {path}")

        return self.model


class EqualWeightAgent:

    def __init__(self, number_of_assets):

        self.number_of_assets = number_of_assets
        self.equal_weights    = np.ones(number_of_assets, dtype=np.float32) / number_of_assets


    def predict(self, observation, deterministic=True):

        return self.equal_weights.copy(), None