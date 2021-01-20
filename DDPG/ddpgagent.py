from time import time

from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
from DDPG.model import ActorCritic


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


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, action_limit: int, epochs_num: int, horizon_len: int,
                 timesteps_per_epoch: int, actor_lr: float, critic_lr: float,
                 start_steps: int, action_noise: float, max_ep_length: int, update_after: int, update_every: int,
                 update_times: int, batch_size: int, gamma: float, polyak: float, activation=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.epochs_num = epochs_num
        self.timesteps_per_epoch = timesteps_per_epoch
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

    def run(self, env, path, epoch):
        starting_timestep = 1
        if path is not None:
            self.actor_critic.load_state_dict(torch.load(path))
            starting_timestep = epoch * self.timesteps_per_epoch

        total_steps = self.epochs_num * self.timesteps_per_epoch
        state, ep_ret, ep_len = env.reset(), 0, 0
        done = self.memory.get()
        loss_q_old, loss_pi_old = self.compute_loss_q(done), self.compute_loss_pi(done)
        start_time = time()

        for t in range(starting_timestep, total_steps):
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

            if (t + 1) % self.timesteps_per_epoch == 0:
                epoch = (t + 1) // self.timesteps_per_epoch

                if (epoch % 50 == 0) or (epoch == self.epochs_num):
                    ac_path = 'output_models/ddpg/actor_critic_{}.pkl'.format(epoch)
                    torch.save(self.actor_critic.state_dict(), ac_path)

                data = self.memory.get()
                loss_q, loss_pi = self.compute_loss_q(data), self.compute_loss_pi(data)
                print("*" * 50 + 'EPOCH {}'.format(epoch) + "*" * 50)
                print('Q loss: {:.6f}, Pi loss : {:.6f}, Delta loss q: {:.6f} Delta loss pi: {:.6f}'.format(
                    loss_q, loss_pi, loss_q - loss_q_old, loss_pi - loss_pi_old))
                print("Exec time: {:.3f}s".format(time() - start_time))
                loss_q_old, loss_pi_old = loss_q, loss_pi
                start_time = time()

    def play(self, env, actor_critic_model_path):
        self.actor_critic.load_state_dict(torch.load(actor_critic_model_path))

        for i in range(self.epochs_num):
            o, ep_ret, ep_len = env.reset(), 0, 0
            for j in range(self.timesteps_per_epoch):
                a = self.get_action(o, self.action_noise)
                o2, r, d, _ = env.step(a)
                o = o2
                env.render()
                d = False if ep_len == self.max_ep_length else d
                if d:
                    break

    def play(self, env, actor_critic_model_path):
        self.actor_critic.load_state_dict(torch.load(actor_critic_model_path))

        for i in range(self.epochs_num):
            state, ep_return, ep_length = env.reset(), 0, 0
            for j in range(self.timesteps_per_epoch):
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
