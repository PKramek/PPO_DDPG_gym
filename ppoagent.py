import logging
from time import time

import numpy as np
import torch
from scipy import signal
from torch import optim

from model import Actor, Critic


class PPOMemory:
    def __init__(self, state_dim, action_dim, buffer_size, gamma, lambda_):
        self.state_memory = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.advantage_memory = np.zeros(buffer_size, dtype=np.float32)
        self.reward_memory = np.zeros(buffer_size, dtype=np.float32)
        self.ret_memory = np.zeros(buffer_size, dtype=np.float32)
        self.value_memory = np.zeros(buffer_size, dtype=np.float32)
        self.log_prob_memory = np.zeros(buffer_size, dtype=np.float32)

        self.gamma = gamma
        self.lambda_ = lambda_

        self.index = 0
        self.path_start_index = 0
        self.max_size = buffer_size

    def store(self, state, action, reward, value, log_prob):
        assert self.index < self.max_size

        self.state_memory[self.index] = state
        self.action_memory[self.index] = action
        self.reward_memory[self.index] = reward
        self.value_memory[self.index] = value
        self.log_prob_memory[self.index] = log_prob

        self.index += 1

    def finish_trajectory(self, current_value: float = 0):
        path_slice = slice(self.path_start_index, self.index)

        values = np.append(self.value_memory[path_slice], current_value)
        rewards = np.append(self.reward_memory[path_slice], current_value)
        print('Mean rewards: {}'.format(np.mean(rewards)))

        delta_t = rewards[:-1] + self.gamma * values[:-1] - values[:-1]  # equation 12 from paper
        # TODO check if it works
        self.advantage_memory[path_slice] = self.discouted_cumulative_sum(delta_t, self.lambda_ * self.gamma)
        self.ret_memory[path_slice] = self.discouted_cumulative_sum(rewards, self.gamma)[:-1]

        self.path_start_index = self.index

    def get(self):
        assert self.index == self.max_size
        self.index = 0
        self.path_start_index = 0

        self._normalize_advantage()
        data = dict(observations=self.state_memory, actions=self.action_memory, rewards=self.reward_memory,
                    ret=self.ret_memory, advantages=self.advantage_memory, log_probabilities=self.log_prob_memory)

        return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in data.items()}

    def discouted_cumulative_sum(self, vector, discount: float):
        # source: https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation
        v = vector[::-1]
        a = [1, -discount]
        b = [1]
        y = signal.lfilter(b, a, x=v)
        return y[::-1]

    def _normalize_advantage(self):
        numpy_advantages = np.array(self.advantage_memory, dtype=np.float32)

        adv_sum = np.sum(numpy_advantages)
        adv_len = len(numpy_advantages)

        mean_advantage = adv_sum / adv_len
        squared_sqrt = (numpy_advantages - mean_advantage) ** 2
        std_advantage = np.sum(squared_sqrt) / adv_len

        self.advantage_memory = (self.advantage_memory - mean_advantage) / std_advantage


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, epochs_num: int, horizon_len: int, max_epoch_len: int,
                 actor_lr: float, critic_lr: float, gamma: float, lambda_: float, epsilon: float,
                 actor_activation=None, critic_activation=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epochs_num = epochs_num
        self.max_epoch_len = max_epoch_len
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.horizon_len = horizon_len

        # TODO make this constructor parameter
        self.train_actor_iterations = 80
        self.train_critic_iterations = 80

        self.actor = Actor(state_dim, action_dim, actor_activation)
        self.critic = Critic(state_dim, critic_activation)

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            # self.actor.cuda(self.device)
            # self.critic.cuda(self.device)
        else:
            self.device = torch.device('cpu')

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.memory = PPOMemory(state_dim, action_dim, horizon_len, gamma, lambda_, )

    def actor_loss(self, data):
        observations = data['observations']
        actions = data['actions']
        advantages = data['advantages']
        log_probabilities_old = data['log_probabilities']

        pi, log_probabilities = self.actor(observations, actions)
        r = torch.exp(log_probabilities - log_probabilities_old)

        clipped_advantage = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon) * advantages
        # TODO check why minus?
        actor_loss = -(torch.min(r * advantages, clipped_advantage)).mean()

        return actor_loss

    def critic_loss(self, data):
        observations = data['observations']
        # TODO change this name: probably to expected returns
        ret = data['ret']

        critic_loss = ((self.critic(observations) - ret) ** 2).mean()
        return critic_loss

    def update(self):
        data = self.memory.get()

        for i in range(self.train_actor_iterations):
            self.actor_optimizer.zero_grad()
            actor_loss = self.actor_loss(data)
            actor_loss.backward()
            self.actor_optimizer.step()

        for i in range(self.train_critic_iterations):
            self.critic_optimizer.zero_grad()
            critic_loss = self.critic_loss(data)
            critic_loss.backward()
            self.critic_optimizer.step()

    def actor_critic_step(self, state):
        with torch.no_grad():
            actor_policy = self.actor.distribution(state)
            action = actor_policy.sample()
            log_probability = self.actor.get_log_probabilities(actor_policy, action)
            value = self.critic(state)

        return action.numpy(), value.numpy(), log_probability.numpy()

    def run(self, env):
        start_time = time()
        state, ep_return, ep_length = env.reset(), 0, 0

        for epoch in range(self.epochs_num):
            for i in range(self.horizon_len):
                action, value, log_probability = self.actor_critic_step(torch.as_tensor(state, dtype=torch.float32))

                next_state, reward, done, _ = env.step(action)
                # env.render()

                ep_return += reward
                ep_length += 1

                self.memory.store(state, action, reward, value, log_probability)

                state = next_state

                timeout = ep_length == self.max_epoch_len
                terminal = done or timeout
                epoch_ended = i == self.horizon_len - 1

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_length, flush=True)
                    if timeout or epoch_ended:
                        _, v, _ = self.actor_critic_step(torch.as_tensor(state, dtype=torch.float32))
                    else:
                        v = 0
                    self.memory.finish_trajectory(v)
                    state, ep_return, ep_length = env.reset(), 0, 0

            self.update()
        env.close()
        duration = time() - start_time
        logging.info('exec time: {}'.format(duration))
