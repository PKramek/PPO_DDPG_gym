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

        rewards = np.append(self.reward_memory[path_slice], current_value)
        values = np.append(self.value_memory[path_slice], current_value)

        delta_t = rewards[:-1] + self.gamma * values[1:] - values[:-1]  # equation 12 from paper

        self.advantage_memory[path_slice] = self.discouted_cumulative_sum(delta_t, self.gamma * self.lambda_)
        self.ret_memory[path_slice] = self.discouted_cumulative_sum(rewards, self.gamma)[:-1]

        self.path_start_index = self.index

    def get(self):
        assert self.index == self.max_size
        self.index = 0
        self.path_start_index = 0

        self._normalize_advantage()

        data = dict(observations=self.state_memory, actions=self.action_memory, ret=self.ret_memory,
                    advantages=self.advantage_memory, log_probabilities=self.log_prob_memory)

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
        advantages_mean = np.mean(numpy_advantages)
        advantages_sd = np.std(numpy_advantages)

        self.advantage_memory = (self.advantage_memory - advantages_mean) / advantages_sd

    def get_mean_reward(self):
        return np.mean(self.reward_memory)

    def get_mean_advantage(self):
        return np.mean(self.advantage_memory)


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, epochs_num: int, horizon_len: int, max_epoch_len: int,
                 actor_lr: float, critic_lr: float, actor_train_iter: int, critic_train_iter, gamma: float,
                 lambda_: float, epsilon: float,
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

        self.train_actor_iterations = actor_train_iter
        self.train_critic_iterations = critic_train_iter

        self.actor = Actor(state_dim, action_dim, actor_activation)
        self.critic = Critic(state_dim, critic_activation)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.memory = PPOMemory(state_dim, action_dim, horizon_len, gamma, lambda_, )

        self.minibatch_size = 64

    def actor_loss(self, data, minibatch_indexes):
        observations = data['observations'][minibatch_indexes]
        actions = data['actions'][minibatch_indexes]
        advantages = data['advantages'][minibatch_indexes]
        log_probabilities_old = data['log_probabilities'][minibatch_indexes]

        pi, log_probabilities = self.actor(observations, actions)
        r = torch.exp(log_probabilities - log_probabilities_old)

        clipped_advantage = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon) * advantages
        # TODO check why minus?
        actor_loss = -(torch.min(r * advantages, clipped_advantage)).mean()

        return actor_loss

    def critic_loss(self, data, minibatch_indexes):
        observations = data['observations'][minibatch_indexes]
        ret = data['ret'][minibatch_indexes]

        estimates = self.critic(observations)
        critic_loss = ((estimates - ret) ** 2).mean()
        return critic_loss

    def get_minibatch_indicies(self):
        return np.random.choice(self.horizon_len, self.minibatch_size)

    def update(self):
        data = self.memory.get()

        for i in range(self.train_actor_iterations):
            minibatch_indexes = self.get_minibatch_indicies()
            self.actor_optimizer.zero_grad()
            actor_loss = self.actor_loss(data, minibatch_indexes)
            actor_loss.backward()
            self.actor_optimizer.step()

        for i in range(self.train_critic_iterations):
            minibatch_indexes = self.get_minibatch_indicies()
            self.critic_optimizer.zero_grad()
            critic_loss = self.critic_loss(data, minibatch_indexes)
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
        prev_episode_return = 0
        state, ep_return, ep_length = env.reset(), 0, 0

        for epoch in range(self.epochs_num):
            print('Update: {}, mean reward = {:.3f}, episode return = {:.3f}'.format(
                epoch, self.memory.get_mean_reward(), prev_episode_return))
            for i in range(self.horizon_len):
                action, value, log_probability = self.actor_critic_step(torch.as_tensor(state, dtype=torch.float32))

                next_state, reward, done, _ = env.step(action)
                env.render()

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
                        pass
                    if timeout or epoch_ended:
                        _, v, _ = self.actor_critic_step(torch.as_tensor(state, dtype=torch.float32))
                    else:
                        v = 0
                    self.memory.finish_trajectory(v)
                    prev_episode_return = ep_return
                    state, ep_return, ep_length = env.reset(), 0, 0

            self.update()

            if epoch % 500 == 0:
                actor_path = 'output_models/actor_{}.pkl'.format(epoch)
                critic_path = 'output_models/critic_{}.pkl'.format(epoch)

                torch.save(self.actor.model, actor_path)
                torch.save(self.critic.model, critic_path)

        env.close()
        duration = time() - start_time
        print('exec time: {}'.format(duration))

    def play(self, env, actor_model_path, critic_model_path):
        self.actor.model = torch.load(actor_model_path)
        self.critic.model = torch.load(critic_model_path)

        for i in range(self.epochs_num):
            state, ep_return, ep_length = env.reset(), 0, 0
            for j in range(self.max_epoch_len):
                action, value, log_probability = self.actor_critic_step(torch.as_tensor(state, dtype=torch.float32))
                next_state, reward, done, _ = env.step(action)
                env.render()
