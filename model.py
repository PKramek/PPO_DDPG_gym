import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, activation=None):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        if activation is None:
            activation = nn.ReLU

        self.model = nn.Sequential(
            nn.Linear(in_features=self.state_dim, out_features=64),
            activation(),
            nn.Linear(in_features=64, out_features=64),
            activation(),
            nn.Linear(in_features=64, out_features=self.action_dim)
        )

    def forward(self, state, action=None):
        pi = self._distribution(state)
        log_prob = None

        if action is not None:
            log_prob = self.get_log_probabilities(pi, action)

        return pi, log_prob

    def get_log_probabilities(self, pi, action):
        # TODO check what sum does
        return pi.log_prob(action).sum(axis=-1)

    def _distribution(self, state):
        mean = self.model(state)
        std = torch.exp(self.log_std)

        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, state_dim, activation=None):
        super(Critic, self).__init__()

        self.state_dim = state_dim

        if activation is None:
            activation = nn.ReLU

        self.model = nn.Sequential(
            nn.Linear(in_features=self.state_dim, out_features=64),
            activation(),
            nn.Linear(in_features=64, out_features=64),
            activation(),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, state):
        return torch.squeeze(self.model(state), -1)
