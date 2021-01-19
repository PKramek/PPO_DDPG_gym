from abc import ABC


class Agent(ABC):

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def play(self, *args, **kwargs):
        raise NotImplementedError()

    def plot_episode_returns(self, *args, **kwargs):
        raise NotImplementedError()

    def from_config_file(self, *args, **kwargs):
        raise NotImplementedError()

    def get_avg_episode_return(self, *args, **kwargs):
        raise NotImplementedError()
