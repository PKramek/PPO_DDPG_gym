import torch

from model import Actor, Critic


class Agent:
    def __init__(self, state_dim: int, action_dim: int, num_iter: int, num_steps: int, learning_rate: float,
                 gamma: float, horizon_len: int, actor_activation=None, critic_activation=None):
        self.num_iter = num_iter
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.horizon_len = horizon_len

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.actor = Actor(state_dim, action_dim, actor_activation)
        self.critic = Critic(state_dim, critic_activation)
