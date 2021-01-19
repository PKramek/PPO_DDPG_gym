from abc import ABC, abstractmethod


class Agent(ABC):

    @classmethod
    @abstractmethod
    def from_config_file(cls, config_file_path, section, state_dim, action_dim):
        raise NotImplementedError()

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def play(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def plot_episode_returns(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_avg_episode_return(self, *args, **kwargs):
        raise NotImplementedError()
