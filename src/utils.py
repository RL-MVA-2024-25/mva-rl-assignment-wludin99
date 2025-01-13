import numpy as np

# Environment wrappers
from gymnasium.wrappers import (
    FrameStack,
    TimeLimit, 
    TransformObservation, 
    TransformReward
)

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from env_hiv_fast import FastHIVPatient

def build_env(randomization, max_steps=200, parallel=False, n_envs=1):
    """creates possibly parallel/multiple env instances.
    log transforms observations, scales down rewards, enforces episode termination, stacks observations"""
    def make_env():
        env = FastHIVPatient(domain_randomization=randomization)
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = Monitor(env=env)
        env = TransformObservation(env, 
                                   lambda obs: np.log(np.maximum(1e-6, obs)) 
                                   )
        env =  TransformReward(env, lambda r: r / 1e8)
        env = FrameStack(env, 5)
        return env
    
    if parallel:
        envs = SubprocVecEnv([make_env for _ in range(n_envs)])
    else: 
        envs = DummyVecEnv([make_env for _ in range(n_envs)])

    return envs

def score(random_reward, deterministic_reward):
    """Computes score in order to determine which models to save"""
    score = 0
    if deterministic_reward >= 3432807.680391572:
        score += 1
    if deterministic_reward >= 1e8:
        score += 1
    if deterministic_reward >= 1e9:
        score += 1
    if deterministic_reward >= 1e10:
        score += 1
    if deterministic_reward >= 2e10:
        score += 1
    if deterministic_reward >= 5e10:
        score += 1
    if random_reward >= 1e10:
        score += 1
    if random_reward >= 2e10:
        score += 1
    if random_reward >= 5e10:
        score += 1
    return score