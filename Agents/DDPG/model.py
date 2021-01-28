import numpy as np

import torch
import torch.nn as nn


def ml_perceptron(sizes, activation, output_activation):
    layers = []
    for j in range(len(sizes)-1):
        action = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), action()]
    return nn.Sequential(*layers)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation, action_limit):
        super().__init__()
        pi_sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        self.pi = ml_perceptron(pi_sizes, activation, nn.Tanh)
        self.action_limit = action_limit

    def forward(self, state):
        return self.action_limit * self.pi(state)


class QFunction(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.q = ml_perceptron([state_dim + action_dim] + list(hidden_sizes) + [1], activation, nn.Identity)

    def forward(self, state, action):
        q = self.q(torch.cat([state, action], dim=-1))
        return torch.squeeze(q, -1)


class ActorCritic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, action_limit: int, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        if activation is None:
            activation = nn.ReLU

        self.pi = Actor(state_dim, action_dim, hidden_sizes, activation, action_limit)
        self.q = QFunction(state_dim, action_dim, hidden_sizes, activation)

    def act(self, state):
        with torch.no_grad():
            return self.pi(state).numpy()
