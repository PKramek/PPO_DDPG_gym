import configparser
from time import time

import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
from Agents.Agent import Agent
from Agents.DDPG.model import ActorCritic


class DDPGMemory:

    def __init__(self, state_dim, action_dim, buffer_size):
        self.state_memory = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(buffer_size, dtype=np.float32)
        self.done_memory = np.zeros(buffer_size, dtype=np.float32)
        self.index, self.size, self.max_size = 0, 0, buffer_size

    def store(self, state, action, reward, next_state, done):
        self.state_memory[self.index] = state
        self.next_state_memory[self.index] = next_state
        self.action_memory[self.index] = action
        self.reward_memory[self.index] = reward
        self.done_memory[self.index] = done
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        indexes = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.state_memory[indexes],
                     next_state=self.next_state_memory[indexes],
                     action=self.action_memory[indexes],
                     reward=self.reward_memory[indexes],
                     done=self.done_memory[indexes])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def get(self):
        indexes = range(self.size)
        batch = dict(state=self.state_memory[indexes],
                     next_state=self.next_state_memory[indexes],
                     action=self.action_memory[indexes],
                     reward=self.reward_memory[indexes],
                     done=self.done_memory[indexes])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class DDPGAgent(Agent):
    def __init__(self, state_dim: int, action_dim: int, action_limit: int, epochs_num: int, horizon_len: int,
                 episodes_per_epoch: int, timesteps_per_episode: int, actor_lr: float, critic_lr: float,
                 start_steps: int, action_noise: float, max_ep_length: int, update_after: int, update_every: int,
                 update_times: int, batch_size: int, gamma: float, polyak: float, activation=None,
                 device=None, benchmark_interval: int = 10, save_model_interval=50):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.epochs_num = epochs_num
        self.episodes_per_epoch = episodes_per_epoch
        self.timesteps_per_episode = timesteps_per_episode
        self.start_steps = start_steps
        self.action_noise = action_noise
        self.max_ep_length = max_ep_length
        self.update_after = update_after
        self.update_every = update_every
        self.update_times = update_times
        self.batch_size = batch_size
        self.gamma = gamma
        self.polyak = polyak

        self.actor_critic = ActorCritic(state_dim, action_dim, action_limit, activation=activation)
        self.ac_target = deepcopy(self.actor_critic)

        for p in self.ac_target.parameters():
            p.requires_grad = False

        self.memory = DDPGMemory(state_dim, action_dim, horizon_len)

        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=actor_lr)
        self.q_optimizer = Adam(self.actor_critic.q.parameters(), lr=critic_lr)

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.avg_episode_returns = []
        self.benchmark_interval = benchmark_interval
        self.save_model_interval = save_model_interval

    @classmethod
    def from_config_file(cls, config_file_path, section, state_dim, action_dim, action_limit, timesteps_per_episode):

        config_file = configparser.ConfigParser()
        config_file.read(config_file_path)

        epochs_num = config_file.getint(section, 'epochs_num')
        horizon_len = config_file.getint(section, 'horizon_len')
        episodes_in_epoch = config_file.getint(section, 'episodes_in_epoch')
        actor_lr = config_file.getfloat(section, 'actor_lr')
        critic_lr = config_file.getfloat(section, 'critic_lr')
        start_steps = config_file.getint(section, 'start_steps')
        action_noise = config_file.getfloat(section, 'action_noise')
        max_ep_length = config_file.getint(section, 'max_ep_length')
        update_after = config_file.getint(section, 'update_after')
        update_every = config_file.getint(section, 'update_every')
        update_times = config_file.getint(section, 'update_times')
        batch_size = config_file.getint(section, 'batch_size')
        gamma = config_file.getfloat(section, 'gamma')
        polyak = config_file.getfloat(section, 'polyak')

        return cls(
            state_dim, action_dim, action_limit, epochs_num, horizon_len, episodes_in_epoch, timesteps_per_episode, actor_lr, critic_lr,
            start_steps, action_noise, max_ep_length, update_after, update_every, update_times, batch_size,
            gamma, polyak)

    def get_action(self, state, noise_scale):
        a = self.actor_critic.act(torch.as_tensor(state, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.action_dim)
        return np.clip(a, -self.action_limit, self.action_limit)

    def compute_loss_q(self, data):
        o, o2, a, r, d = data['state'], data['next_state'], data['action'], data['reward'], data['done']

        q = self.actor_critic.q(o, a)

        with torch.no_grad():
            q_pi_targ = self.ac_target.q(o2, self.ac_target.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        loss_q = ((q - backup) ** 2).mean()

        return loss_q

    def compute_loss_pi(self, data):
        o = data['state']
        q_pi = self.actor_critic.q(o, self.actor_critic.pi(o))
        return -q_pi.mean()

    def update(self, data):
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.actor_critic.q.parameters():
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.actor_critic.q.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.parameters(), self.ac_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_avg_episode_return(self, env, n=10):
        print(f'Calculating average episode return...')
        episodes_return = np.ones(n, dtype=np.float32)
        with torch.no_grad():
            for i in range(n):
                state, ep_return, ep_length = env.reset(), 0, 0
                for timestep in range(self.timesteps_per_episode):
                    state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                    action = self.get_action(state, self.action_noise)

                    state, reward, done, _ = env.step(action)
                    ep_return += reward
                    ep_length += 1

                    done = False if ep_length == self.max_ep_length else done

                    if done:
                        break

                episodes_return[i] = ep_return

        avg_return = np.mean(episodes_return)

        print(f'Average episode return = {avg_return: 3f}\n')

        return avg_return

    def train(self, env):
        timesteps_per_epoch = self.timesteps_per_episode * self.episodes_per_epoch
        total_steps = self.epochs_num * timesteps_per_epoch
        state, ep_ret, ep_len = env.reset(), 0, 0
        done = self.memory.get()
        loss_q_old, loss_pi_old = self.compute_loss_q(done), self.compute_loss_pi(done)
        env_name = env.spec.id
        start_time = time()

        print("*" * 50 + 'EPOCH 1' + "*" * 50)
        for t in range(total_steps):
            if t > self.start_steps:
                action = self.get_action(state, self.action_noise)
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            ep_ret += reward
            ep_len += 1

            done = False if ep_len == self.max_ep_length else done

            self.memory.store(state, action, reward, next_state, done)

            state = next_state

            if done or (ep_len == self.max_ep_length):
                print("EpRet: {}, EpLen: {}".format(ep_ret, ep_len))
                state, ep_ret, ep_len = env.reset(), 0, 0

            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.memory.sample_batch(self.batch_size)
                    self.update(data=batch)

            if (t + 1) % timesteps_per_epoch == 0:
                epoch = (t + 1) // timesteps_per_epoch

                if epoch % self.benchmark_interval == 0:
                    self.avg_episode_returns.append(self.get_avg_episode_return(env))
                if epoch % self.save_model_interval == 0:
                    ac_path = 'trained_models/DDPG/{}_actor_critic_{}_epochs.pkl'.format(env_name, epoch)
                    torch.save(self.actor_critic.state_dict(), ac_path)

                data = self.memory.get()
                loss_q, loss_pi = self.compute_loss_q(data), self.compute_loss_pi(data)
                print('Q loss: {:.6f}, Pi loss : {:.6f}, Delta loss q: {:.6f} Delta loss pi: {:.6f}'.format(
                    loss_q, loss_pi, loss_q - loss_q_old, loss_pi - loss_pi_old))
                print("Exec time: {:.3f}s".format(time() - start_time))
                print("*" * 50 + 'EPOCH {}'.format(epoch+1) + "*" * 50)
                loss_q_old, loss_pi_old = loss_q, loss_pi
                start_time = time()

    def play(self, env, actor_critic_model_path):
        self.actor_critic.load_state_dict(torch.load(actor_critic_model_path))

        for i in range(self.epochs_num):
            state, ep_return, ep_length = env.reset(), 0, 0
            for j in range(self.timesteps_per_episode * self.episodes_per_epoch):
                action = self.get_action(state, self.action_noise)

                next_state, reward, done, _ = env.step(action)
                ep_return += reward
                ep_length += 1

                done = False if ep_length == self.max_ep_length else done

                self.memory.store(state, action, reward, next_state, done)

                state = next_state
                env.render()
                if done:
                    break
            self.avg_episode_returns.append(self.get_avg_episode_return(env))

    def plot_episode_returns(self, path=None):
        x_axis = [x * self.benchmark_interval * self.timesteps_per_episode * self.episodes_per_epoch for x in range(len(self.avg_episode_returns))]
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