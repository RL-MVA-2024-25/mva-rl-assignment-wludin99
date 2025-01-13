from pathlib import Path
import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from coolname import generate_slug

try:    
    import wandb
    use_wandb = True
except:
    use_wandb = False

from utils import build_env
from utils import score as utils_score

class EvalCallback(BaseCallback):
    def __init__(self, n_episodes=50, experiment_name=generate_slug()):
        super().__init__(verbose=1)

        self.best_score = self.best_deterministic_reward = self.best_random_reward = float("-inf")
        self.models_path = Path(__file__).parent.parent / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.n_episodes = n_episodes

        self.experiment_name = experiment_name

    def _on_step(self):
        random_env = build_env(randomization=True)
        deterministic_env = build_env(randomization=False)

        mean_reward_random, std_reward_random = evaluate_policy(
            self.model, random_env, n_eval_episodes=self.n_episodes
        )
        mean_reward_deterministic, std_reward_deterministic = evaluate_policy(
            self.model, deterministic_env, n_eval_episodes=self.n_episodes
        )

        # save model if it's the best so far
        score = utils_score(mean_reward_random * 1e8, mean_reward_deterministic * 1e8)
        if score >= self.best_score or (
            mean_reward_random > self.best_random_reward
            and mean_reward_deterministic > self.best_deterministic_reward
        ):
            
            self.model.save(
                os.path.join(
                    self.models_path,
                    self.experiment_name,
                    "best_model.zip"
                )
            )

            # log to wandb
            if use_wandb:
                wandb.log(
                {
                    "eval/score": score,
                    "eval/mean_reward_random": mean_reward_random,
                    "eval/std_reward_random": std_reward_random,
                    "eval/mean_reward_deterministic": mean_reward_deterministic,
                    "eval/std_reward_deterministic": std_reward_deterministic,
                }
        )
        
        return True