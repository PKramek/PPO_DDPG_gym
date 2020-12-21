import gym
import torch

from ppoagent import PPOAgent

assert torch.cuda.is_available()

gym.envs.register(
    id='DoublePrecisionSwimmer-v2',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    max_episode_steps=2000
)


def cut_timestep_in_half(env):
    env.model.opt.timestep = env.model.opt.timestep / 2


env = gym.make('HalfCheetah-v2')

horizon_len = 4000
timesteps_per_epoch = 1000
max_timesteps_per_epoch = 1000
epochs = 2500
gamma = 0.99
epsilon = 0.2
lambda_ = 0.97
actor_learning_rate = 3e-4
critic_learning_rate = 1e-3
train_actor_iterations = 80
train_critic_iterations = 80
minibatch_size = 64

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPOAgent(
    state_dim, action_dim, epochs, horizon_len, timesteps_per_epoch, max_timesteps_per_epoch, actor_learning_rate,
    critic_learning_rate, train_actor_iterations, train_critic_iterations, minibatch_size, gamma, lambda_, epsilon)
print(agent.actor)
print(agent.critic)

agent.run(env)


