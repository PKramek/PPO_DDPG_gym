from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from torch import optim

from Agents.Agent import Agent
from Agents.PPO.model import Actor, Critic


class PPOMemory:
    def __init__(self, state_dim, action_dim, buffer_size, gamma, lambda_, device=None):
        if device is None:
            device = torch.device('cpu')

        self.state_memory = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.action_memory = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.advantage_memory = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.reward_memory = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.discounted_rewards_memory = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.value_memory = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_prob_memory = torch.zeros(buffer_size, dtype=torch.float32, device=device)

        self.gamma = gamma
        self.lambda_ = lambda_

        self.index = 0
        self.path_start_index = 0
        self.max_size = buffer_size

        self.device = device

    def store(self, state, action, reward, value, log_prob):
        assert self.index < self.max_size

        self.state_memory[self.index] = state
        self.action_memory[self.index] = action
        self.reward_memory[self.index] = reward
        self.value_memory[self.index] = value
        self.log_prob_memory[self.index] = log_prob

        self.index += 1

    def end_episode(self, current_value: float = 0):
        path_slice = slice(self.path_start_index, self.index)
        current_value_tensor = torch.tensor([current_value], device=self.device)
        rewards = torch.cat((self.reward_memory[path_slice], current_value_tensor))
        values = torch.cat((self.value_memory[path_slice], current_value_tensor))

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]  # equation 12 from paper
        advantages = self.discouted_cumulative_sum(
            deltas.cpu().numpy(), self.gamma * self.lambda_)
        # copy() must be used because advantage numpy array has negative stride:
        # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
        advantages = torch.as_tensor(advantages.copy(), dtype=torch.float32, device=self.device)

        discounted_rewards = self.discouted_cumulative_sum(rewards.cpu().numpy(), self.gamma)[:-1]
        discounted_rewards = torch.as_tensor(discounted_rewards.copy(), dtype=torch.float32, device=self.device)

        self.advantage_memory[path_slice] = advantages
        self.discounted_rewards_memory[path_slice] = discounted_rewards

        self.path_start_index = self.index

    def get(self):
        assert self.index == self.max_size
        self.index = 0
        self.path_start_index = 0

        data = dict(states=self.state_memory, actions=self.action_memory,
                    discounted_rewards=self.discounted_rewards_memory,
                    advantages=self.advantage_memory, log_probabilities=self.log_prob_memory)

        return {key: torch.as_tensor(value, dtype=torch.float32, device=self.device) for key, value in data.items()}

    def discouted_cumulative_sum(self, vector, discount: float):
        # source: https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation
        v = vector[::-1]
        a = [1, -discount]
        b = [1]
        y = signal.lfilter(b, a, x=v)
        return y[::-1]


class PPOAgent(Agent):
    def __init__(self, state_dim: int, action_dim: int, epochs_num: int, horizon_len: int, timesteps_per_epoch: int,
                 actor_lr: float, critic_lr: float, actor_train_iter: int, critic_train_iter: int, minibatch_size: int,
                 gamma: float, lambda_: float, epsilon: float, actor_activation=None, critic_activation=None,
                 device=None, hidden_size=64, benchmark_interval=10, save_model_interval = 250):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epochs_num = epochs_num
        self.timesteps_per_epoch = timesteps_per_epoch
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.horizon_len = horizon_len

        self.train_actor_iterations = actor_train_iter
        self.train_critic_iterations = critic_train_iter

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.actor = Actor(state_dim, action_dim, actor_activation, self.device, hidden_size)
        self.critic = Critic(state_dim, critic_activation, self.device, hidden_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.memory = PPOMemory(state_dim, action_dim, horizon_len, gamma, lambda_, self.device)

        self.minibatch_size = minibatch_size

        self.avg_episode_returns = []
        self.benchmark_interval = benchmark_interval
        self.save_model_interval = save_model_interval

    def actor_loss(self, data, minibatch_indexes):
        states = data['states'][minibatch_indexes]
        actions = data['actions'][minibatch_indexes]
        advantages = data['advantages'][minibatch_indexes]
        log_probabilities_old = data['log_probabilities'][minibatch_indexes]

        pi, log_probabilities = self.actor(states, actions)
        ratio = torch.exp(log_probabilities - log_probabilities_old)

        clipped_advantage = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        ratio_times_advantage = ratio * advantages

        actor_loss = - torch.mean(torch.min(ratio_times_advantage, clipped_advantage))

        return actor_loss

    def critic_loss(self, data, minibatch_indexes):
        observations = data['states'][minibatch_indexes]
        ret = data['discounted_rewards'][minibatch_indexes]

        critic_loss = ((self.critic(observations) - ret) ** 2).mean()

        return critic_loss

    def get_minibatch_indicies(self):
        return np.random.choice(self.horizon_len, self.minibatch_size)

    def update(self):
        data = self.memory.get()
        all_data_indexes = np.arange(self.horizon_len)
        actor_loss_old = self.actor_loss(data, all_data_indexes)
        actor_loss_old = actor_loss_old.item()
        critic_loss_old = self.critic_loss(data, all_data_indexes)

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

        with torch.no_grad():
            actor_loss = self.actor_loss(data, all_data_indexes)
            actor_loss = actor_loss.item()
            critic_loss = self.critic_loss(data, all_data_indexes)
            critic_loss = critic_loss.item()

        print('Actor loss: {:.6f}, Critic loss : {:.6f}, Delta loss pi: {:.6f} Delta loss v: {:.6f}'.format(
            actor_loss_old, critic_loss_old, actor_loss - actor_loss_old,
                                             critic_loss - critic_loss_old))
        print('\n')

    def actor_critic_step(self, state):
        with torch.no_grad():
            actor_policy = self.actor.distribution(state)
            action = actor_policy.sample()
            log_probability = self.actor.get_log_probabilities(actor_policy, action)
            value = self.critic(state)

        return action, value, log_probability

    def get_avg_episode_return(self, env, n=10):
        print(f'Inside get_avg_episode_return')
        episodes_return = np.ones(n, dtype=np.float32)
        with torch.no_grad():
            for i in range(n):
                state, ep_return = env.reset(), 0
                for timestep in range(self.timesteps_per_epoch):
                    state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                    action, _, _ = self.actor_critic_step(state)

                    action_cpu = action.cpu()
                    state, reward, done, _ = env.step(action_cpu)

                    ep_return += reward

                    if done:
                        break
                print(f'{i} episode return = {ep_return}')
                episodes_return[i] = ep_return

        return np.mean(episodes_return)

    def train(self, env):
        start_time = time()
        state, ep_return, timestep_in_horizon = env.reset(), 0, 0

        self.avg_episode_returns = []

        for epoch in range(1, self.epochs_num + 1):
            epoch_start_time = time()
            print("*" * 50 + 'EPOCH {}'.format(epoch) + "*" * 50)

            for timestep in range(self.horizon_len):
                state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                action, value, log_probability = self.actor_critic_step(state)

                # GPU tensors can not be converted to numpy arrays
                action_cpu = action.cpu()
                next_state, reward, done, _ = env.step(action_cpu)

                ep_return += reward
                timestep_in_horizon += 1

                self.memory.store(state, action, reward, value, log_probability)

                state = next_state
                timeout = timestep_in_horizon == self.timesteps_per_epoch
                is_terminal = done or timeout
                epoch_ended = timestep == self.horizon_len - 1

                if is_terminal or epoch_ended:
                    if epoch_ended and not is_terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.' % timestep_in_horizon, flush=True)
                    if timeout or epoch_ended:
                        _, value, _ = self.actor_critic_step(
                            torch.as_tensor(state, dtype=torch.float32, device=self.device))
                    else:
                        value = 0

                    self.memory.end_episode(value)
                    if is_terminal:
                        print("Episode return: {}, Episode_len: {}, Mean return per step: {}".format(
                            ep_return,
                            timestep_in_horizon,
                            ep_return / float(timestep_in_horizon)))
                    state, ep_return, timestep_in_horizon = env.reset(), 0, 0

            self.update()

            if epoch % self.benchmark_interval == 0:
                self.avg_episode_returns.append(self.get_avg_episode_return(env))

            if epoch % self.save_model_interval == 0:
                actor_path = 'trained_models/actor_{}.pkl'.format(epoch)
                critic_path = 'trained_models/critic_{}.pkl'.format(epoch)
                torch.save(self.actor.state_dict(), actor_path)

                torch.save(self.critic.state_dict(), critic_path)
            print(f'Epoch exec time {time() - epoch_start_time}s')

        print("Exec time: {:.3f}s".format(time() - start_time))

    def play(self, env, actor_model_path, critic_model_path):
        self.actor.load_state_dict(torch.load(actor_model_path))
        self.critic.load_state_dict(torch.load(critic_model_path))

        for i in range(self.epochs_num):
            state, ep_return, ep_length = env.reset(), 0, 0
            for j in range(self.timesteps_per_epoch):
                step = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                action, value, log_probability = self.actor_critic_step(step)
                state, reward, done, _ = env.step(action)

                if done:
                    break

                env.render()

    def plot_episode_returns(self, path=None):
        x_axis = [x * self.benchmark_interval * self.horizon_len for x in range(len(self.avg_episode_returns))]
        plt.plot(x_axis, self.avg_episode_returns)
        plt.xlabel('Timestep')
        plt.ylabel('Avg. episode return')
        plt.grid()

        if path is not None:
            assert isinstance(path, str)
            try:
                plt.savefig(path)
            except IOError:
                print(f'Could not save figure under {path}')

        plt.show()
