from collections import deque
import numpy as np
import random
import torch
from trees import SumTree, MinTree

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, n_step, gamma, stack_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.stack_size = stack_size
        self.n_step_buffer = deque(maxlen=n_step)
        self.frame_stack = deque(maxlen=stack_size)

    def add(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        self.frame_stack.append(state)
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self._get_n_step_info()
            stacked_state = self._get_stacked_state()
            stacked_next_state = self._get_stacked_next_state(next_state)
            self.memory.append((stacked_state, action, reward, stacked_next_state, done))

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[2:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        state, action = self.n_step_buffer[0][:2]
        return state, action, reward, next_state, done

    def _get_stacked_state(self):
        while len(self.frame_stack) < self.stack_size:
            self.frame_stack.appendleft(np.zeros_like(self.frame_stack[0]))
        return np.array(self.frame_stack).flatten()

    def _get_stacked_next_state(self, next_state):
        next_frame_stack = self.frame_stack.copy()
        next_frame_stack.append(next_state)
        while len(next_frame_stack) < self.stack_size:
            next_frame_stack.appendleft(np.zeros_like(next_frame_stack[0]))
        return np.array(next_frame_stack).flatten()

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, buffer_size, batch_size, n_step, gamma, stack_size, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        super().__init__(action_size, buffer_size, batch_size, n_step, gamma, stack_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.sum_tree = SumTree(buffer_size)
        self.min_tree = MinTree(buffer_size)
        self.epsilon = 1e-5

    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self._get_n_step_info()
            stacked_state = self._get_stacked_state()
            stacked_next_state = self._get_stacked_next_state(next_state)
            max_priority = self.sum_tree.tree.max() if self.sum_tree.size > 0 else 1.0
            self.sum_tree.add(max_priority, (stacked_state, action, reward, stacked_next_state, done))
            self.min_tree.add(max_priority, (stacked_state, action, reward, stacked_next_state, done))

    def sample(self):
        experiences = []
        indices = []
        priorities = []
        segment = self.sum_tree.total() / self.batch_size

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(self.batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            idx, priority, data = self.sum_tree.get(s)
            experiences.append(data)
            indices.append(idx)
            priorities.append(priority)

        sampling_probabilities = priorities / self.sum_tree.total()
        is_weights = np.power(self.sum_tree.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()
        is_weights = torch.from_numpy(is_weights).float()

        return (states, actions, rewards, next_states, dones, indices, is_weights)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.sum_tree.update(idx, priority + self.epsilon)
            self.min_tree.update(idx, priority + self.epsilon)