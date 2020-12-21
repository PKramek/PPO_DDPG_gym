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
        self.rewards_to_go_memory = np.zeros(buffer_size, dtype=np.float32)
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

        daltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]  # equation 12 from paper
        self.advantage_memory[path_slice] = self.discouted_cumulative_sum(daltas, self.gamma * self.lambda_)
        self.rewards_to_go_memory[path_slice] = self.discouted_cumulative_sum(rewards, self.gamma)[:-1]

        self.path_start_index = self.index

    def get(self):
        assert self.index == self.max_size
        self.index = 0
        self.path_start_index = 0

        self._normalize_advantage()

        data = dict(states=self.state_memory, actions=self.action_memory, rewards_to_go=self.rewards_to_go_memory,
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
    def __init__(self, state_dim: int, action_dim: int, epochs_num: int, horizon_len: int, timesteps_per_epoch: int,
                 max_timesteps_per_epoch: int, actor_lr: float, critic_lr: float,
                 actor_train_iter: int, critic_train_iter: int, minibatch_size: int,
                 gamma: float, lambda_: float, epsilon: float, actor_activation=None, critic_activation=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epochs_num = epochs_num
        self.timesteps_per_epoch = timesteps_per_epoch  # combined
        self.max_timesteps_per_epoch = max_timesteps_per_epoch  # for a single agent
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

        self.minibatch_size = minibatch_size

        self.target_kl = 0.01

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

        apox_kl = (log_probabilities_old - log_probabilities).mean().item()
        entropy = pi.entropy().mean().item()
        clipped = ratio.lt(1.0 - self.epsilon) | ratio.gt(1.0 + self.epsilon)
        clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=apox_kl, entropy=entropy, cf=clip_fraction)

        return actor_loss, pi_info

    def critic_loss(self, data, minibatch_indexes):
        observations = data['states'][minibatch_indexes]
        ret = data['rewards_to_go'][minibatch_indexes]

        critic_loss = ((self.critic(observations) - ret) ** 2).mean()

        return critic_loss

    def get_minibatch_indicies(self):
        #TODO check if it should be random or rather series
        return np.random.choice(self.horizon_len, self.minibatch_size)

    def update(self):
        data = self.memory.get()
        all_data_indexes = np.arange(self.horizon_len)
        actor_loss_old, pi_info_old = self.actor_loss(data, all_data_indexes)
        actor_loss_old = actor_loss_old.item()
        critic_loss_old = self.critic_loss(data, all_data_indexes)

        for i in range(self.train_actor_iterations):
            minibatch_indexes = self.get_minibatch_indicies()
            self.actor_optimizer.zero_grad()
            actor_loss, pi_info = self.actor_loss(data, minibatch_indexes)
            kl = pi_info['kl']
            # if kl > 1.5 * self.target_kl:
            #     print('Stopping update, reached max kl {}'.format(i))
            #     break
            actor_loss.backward()
            self.actor_optimizer.step()

        for i in range(self.train_critic_iterations):
            minibatch_indexes = self.get_minibatch_indicies()
            self.critic_optimizer.zero_grad()
            critic_loss = self.critic_loss(data, minibatch_indexes)
            critic_loss.backward()
            self.critic_optimizer.step()

        with torch.no_grad():
            actor_loss, _ = self.actor_loss(data, all_data_indexes)
            actor_loss = actor_loss.item()
            critic_loss = self.critic_loss(data, all_data_indexes)
            critic_loss = critic_loss.item()

        print('Actor loss: {:.6f}, Critic loss : {:.6f},KL: {:.6f}, Delta loss pi: {:.6f} Delta loss v: {:.6f}'.format(
            actor_loss_old, critic_loss_old, 1, actor_loss - actor_loss_old,
                                                critic_loss - critic_loss_old))
        print('\n')

    def actor_critic_step(self, state):
        with torch.no_grad():
            actor_policy = self.actor.distribution(state)
            action = actor_policy.sample()
            log_probability = self.actor.get_log_probabilities(actor_policy, action)
            value = self.critic(state)

        return action.numpy(), value.numpy(), log_probability.numpy()

    def run(self, env):
        start_time = time()
        state, ep_return, timestep_in_horizon = env.reset(), 0, 0
        for epoch in range(1, self.epochs_num + 1):
            print("*" * 50 + 'EPOCH {}'.format(epoch) + "*" * 50)

            for timestep in range(self.horizon_len):
                action, value, log_probability = self.actor_critic_step(torch.as_tensor(state, dtype=torch.float32))
                next_state, reward, done, _ = env.step(action)

                ep_return += reward
                timestep_in_horizon += 1

                self.memory.store(state, action, reward, value, log_probability)

                # Find equation
                state = next_state
                timeout = timestep_in_horizon == self.max_timesteps_per_epoch
                is_terminal = done or timeout
                epoch_ended = timestep == self.horizon_len - 1

                if is_terminal or epoch_ended:
                    if epoch_ended and not is_terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.' % timestep_in_horizon, flush=True)
                    if timeout or epoch_ended:
                        _, value, _ = self.actor_critic_step(torch.as_tensor(state, dtype=torch.float32))
                    else:
                        value = 0

                    self.memory.finish_trajectory(value)
                    if is_terminal:
                        print("Episode return: {}, Episode_len: {}, Mean return per step: {}".format(
                            ep_return,
                            timestep_in_horizon,
                            ep_return / float(timestep_in_horizon)))
                    state, ep_return, timestep_in_horizon = env.reset(), 0, 0

            self.update()

            if epoch % 500 == 0:
                actor_path = 'output_models/actor_{}.pkl'.format(epoch)
                critic_path = 'output_models/critic_{}.pkl'.format(epoch)
                torch.save(self.actor.state_dict(), actor_path)

                torch.save(self.critic.state_dict(), critic_path)

        print("Exec time: {:.3f}s".format(time() - start_time))

    def play(self, env, actor_model_path, critic_model_path):
        self.actor.load_state_dict(torch.load(actor_model_path))
        self.critic.load_state_dict(torch.load(critic_model_path))

        for i in range(self.epochs_num):
            state, ep_return, ep_length = env.reset(), 0, 0
            for j in range(self.timesteps_per_epoch):
                action, value, log_probability = self.actor_critic_step(torch.as_tensor(state, dtype=torch.float32))
                next_state, reward, done, _ = env.step(action)
                state = next_state
                env.render()
                if done:
                    break
