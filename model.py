from torch import nn


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

    def forward(self, state):
        return self.model(state)


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
        self.model(state)
