# Standard imports
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EveryNTimesteps

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    use_wandb = True
except:
    use_wandb = False

# Local imports
from interface import Agent
from callback import EvalCallback
from utils import build_env

# Constants
algos = {
    "ppo": PPO,
    "dqn": DQN
}

# Training function
def train(algo="ppo", timesteps=2e6, n_envs=1, randomization=False):
    config = {
        "num_envs": n_envs,
        "algo": algo,
        "timesteps": timesteps,
        "domain_randomization": randomization
    }

    run = wandb.init(
        config=config,
        project="hiv-rl-assignment",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    
    env = build_env(
        randomization=randomization, 
        parallel=True if n_envs > 1 else False, 
        n_envs=n_envs
    )

    eval_callback = EvalCallback(
        experiment_name=run.name
    )
    spaced_eval_callback = EveryNTimesteps(
        n_steps=int(timesteps / 20),  # every 5% of training
        callback=eval_callback
        )

    model = algos[algo](
        "MlpPolicy",
        env,
        tensorboard_log=f"runs/{run.id}",
        ent_coef=0.2,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64, 64, 64, 64], vf=[64, 64, 64, 64])
        )
    )

    wandb_callback = WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2
            )
    
    callbacks_list = [spaced_eval_callback] + [wandb_callback] if use_wandb else []

    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=callbacks_list
    )

    run.finish()

def main():
    parser = ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2e6,
                       help="Number of training timesteps")
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo",
                       help="RL algorithm used (ppo or dqn)")
    parser.add_argument("--domain-randomization", action="store_true",
                       help="Enable random initialization during training")
    
    args = parser.parse_args()
    train(
        n_envs=4, 
        timesteps=args.timesteps, 
        algo=args.algo,
        randomization=args.domain_randomization
    )

if __name__ == "__main__":
    main()


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

models_path = Path(__file__).parent.parent / "models"
model_name = "ppo/best_model.zip"


class ProjectAgent(Agent):
    def __init__(self):
        self.obs_buffer = None
        self.buffer_size = 5  # Match the stack_size from training
        self.obs_dim = 6  # The dimension of a single observation

        self.policy = None

    def act(self, observation, use_random=False):
        # Transform observation as in training
        observation = np.log(np.maximum(observation, 1e-6))
        
        # Update buffer
        if self.obs_buffer is None:
            self.obs_buffer = np.tile(observation, (self.buffer_size, 1))
        else:
            self.obs_buffer = np.roll(self.obs_buffer, shift=-1, axis=0)
            self.obs_buffer[-1] = observation
        stacked_obs = self.obs_buffer.reshape(self.buffer_size, self.obs_dim)
        
        return self.policy.predict(stacked_obs, deterministic=True)[0]

    def save(self, path):
        self.policy.save(path)

    def load(self):
        self.policy = PPO.load(models_path / model_name)