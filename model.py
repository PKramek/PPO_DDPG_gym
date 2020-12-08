import numpy as np
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

        # TODO check why there is -0.5??
        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, state, action=None):
        pi = self.distribution(state)
        log_prob = None

        if action is not None:
            log_prob = self.get_log_probabilities(pi, action)

        return pi, log_prob

    @staticmethod
    def get_log_probabilities(pi, action):
        # TODO check what sum does
        return pi.log_prob(action).sum(axis=-1)

    def distribution(self, state):
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
        # TODO test what squeeze does
        return torch.squeeze(self.model(state), -1)
