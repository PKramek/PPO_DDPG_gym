import configparser

import gym

from Agents.PPO.ppoagent import PPOAgent

gym.envs.register(
    id='DoublePrecisionSwimmer-v2',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    max_episode_steps=2000
)


def cut_timestep_in_half(env):
    env.model.opt.timestep = env.model.opt.timestep / 2


# env = gym.make('DoublePrecisionSwimmer-v2')
# cut_timestep_in_half(env)

env = gym.make('Swimmer-v2')

horizon_len = 4000
timesteps_per_epoch = 1000
epochs = 250
gamma = 0.99
epsilon = 0.2
lambda_ = 0.95
actor_learning_rate = 1e-3
critic_learning_rate = 1e-3
train_actor_iterations = 150
train_critic_iterations = 150
minibatch_size = 64
hidden_size = 64

config = configparser.ConfigParser()
config['PPO'] = {
    'horizon_length': horizon_len,
    'timesteps_per_epoch': timesteps_per_epoch,
    'epochs': epochs,
    'gamma': gamma,
    'epsilon': epsilon,
    'lambda': lambda_,
    'actor_learning_rate': actor_learning_rate,
    'critic_learning_rate': critic_learning_rate,
    'train_actor_iterations': train_actor_iterations,
    'train_critic_iterations': train_critic_iterations,
    'minibatch_size': minibatch_size,
    'hidden_size': hidden_size
}


with open('example.ini', 'w') as configfile:
    config.write(configfile)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPOAgent.from_config_file('example.ini', state_dim, action_dim)

agent.train(env)
agent.plot_episode_returns('results/PPO.png')
