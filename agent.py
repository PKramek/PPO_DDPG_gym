import numpy as np
import torch
from torch import optim

from model import Actor, Critic


class Memory:
    def __init__(self, state_dim, action_dim, buffer_size, gamma, lambda_):
        self.state_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.advantage_buf = np.zeros(buffer_size, dtype=np.float32)
        self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
        self.advantage_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.value_buf = np.zeros(buffer_size, dtype=np.float32)
        self.log_prob_buf = np.zeros(buffer_size, dtype=np.float32)

        self.gamma = gamma
        self.lambda_ = lambda_

        self.index = 0
        self.path_start_index = 0
        self.max_size = buffer_size

    def store(self, state, action, reward, value, log_prob):
        assert self.index < self.max_size

        self.state_buf[self.index] = state
        self.action_buf[self.index] = action
        self.reward_buf[self.index] = reward
        self.value_buf[self.index] = value
        self.log_prob_buf[self.index] = log_prob

        self.index += 1

    def finish_path(self):
        # TODO finish this method
        pass

    def get(self):
        assert self.index == self.max_size
        self.index = 0

        self._normalize_advantage()
        data = dict(observations=self.state_buf, actions=self.action_buf, rewards=self.reward_buf,
                    ret=self.ret_buf, adventages=self.advantage_buf, log_probabilities=self.log_prob_buf)

        return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in data.items()}

    def _normalize_advantage(self):
        assert self.index == self.max_size

        adv_sum = np.sum(self.advantage_buf)
        adv_len = len(self.advantage_buf)

        mean_advantage = adv_sum / adv_len
        std_advantage = np.sum(np.sqrt(self.advantage_buf - mean_advantage) ** 2) / adv_len

        self.advantage_buf = (self.advantage_buf - mean_advantage) / std_advantage


class Agent:
    def __init__(self, state_dim: int, action_dim: int, num_iter: int, num_steps: int, actor_lr: float,
                 critic_lr: float, gamma: float, horizon_len: int, actor_activation=None, critic_activation=None):
        self.num_iter = num_iter
        self.num_steps = num_steps
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.horizon_len = horizon_len

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.actor = Actor(state_dim, action_dim, actor_activation)
        self.critic = Critic(state_dim, critic_activation)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def actor_loss(self, x, y):

    def critic_loss(self, x, y):
        pass
