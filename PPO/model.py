import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, activation=None, device=None, hidden_size=128):
        super(Actor, self).__init__()

        if device is None:
            device = torch.device('cpu')

        self.state_dim = state_dim
        self.action_dim = action_dim

        if activation is None:
            activation = nn.ReLU

        self.device = device

        self.model = nn.Sequential(
            nn.Linear(in_features=self.state_dim, out_features=hidden_size),
            activation(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            activation(),
            nn.Linear(in_features=hidden_size, out_features=self.action_dim)
        )
        self.std = 0.6 * torch.ones((action_dim,), dtype=torch.float32, device=device)

        self.to(device)
        self.model.to(device)

    def forward(self, state, action=None):
        pi = self.distribution(state)

        log_prob = None
        if action is not None:
            log_prob = self.get_log_probabilities(pi, action)

        return pi, log_prob

    def set_device(self, device):
        self.to(device)
        self.model.to(device)
        self.std.to(device)

    @staticmethod
    def get_log_probabilities(pi, action):
        return pi.log_prob(action).sum(axis=-1)

    def distribution(self, state):
        mean = self.model(state)
        return Normal(mean, self.std)


class Critic(nn.Module):
    def __init__(self, state_dim, activation=None, device='cpu', hidden_size=64):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.device = device

        if activation is None:
            activation = nn.ReLU

        self.model = nn.Sequential(
            nn.Linear(in_features=self.state_dim, out_features=hidden_size),
            activation(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            activation(),
            nn.Linear(in_features=hidden_size, out_features=1)
        )

        self.model.to(device)
        self.to(device)

    def forward(self, state):
        return torch.squeeze(self.model(state), -1)
