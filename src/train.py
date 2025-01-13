import numpy as np
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from env_hiv_fast import FastHIVPatient
import wandb
import yaml
from agent import ProjectAgent

# Load config from file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize wandb
wandb.init(project="rl-class-assignment", config=config)

env = TimeLimit(
    env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# Training loop
state_size = env.observation_space.shape[0] * 4  # Adjust state_size for frame stacking
action_size = env.action_space.n
agent = ProjectAgent(state_size, action_size)

# Log hyperparameters
wandb.config.update(config)

n_episodes = config["n_episodes"]
max_t = config["max_t"]
eps_start = config["eps_start"]
eps_end = config["eps_end"]
eps_decay = config["eps_decay"]
eps = eps_start

for i_episode in range(1, n_episodes + 1):
    state = env.reset()
    state = state[0]  # Unpack the state from the tuple
    agent.memory.frame_stack.clear()
    for _ in range(agent.memory.stack_size - 1):
        agent.memory.frame_stack.append(np.zeros_like(state))
    agent.memory.frame_stack.append(state)
    stacked_state = np.array(agent.memory.frame_stack).flatten()
    total_reward = 0
    eps = max(eps_end, eps_decay * eps)
    for t in range(max_t):
        action = agent.act(stacked_state, eps)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        agent.memory.frame_stack.append(next_state)
        stacked_state = np.array(agent.memory.frame_stack).flatten()
        total_reward += reward
        if done or truncated:
            break
    # Log metrics
    wandb.log({"episode": i_episode, "total_reward": total_reward, "epsilon": eps})
    print(f"Episode {i_episode}/{n_episodes}, Total Reward: {total_reward}")

agent.save("agent.pth")
