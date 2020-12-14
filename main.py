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


env = gym.make('Swimmer-v2')

horizon_len = 2048
max_epoch_len = 2048
steps_per_epoch = 10000
epochs = 10000
gamma = 0.99
epsilon = 0.2
lambda_ = 0.97
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
train_actor_iterations = 50
train_critic_iterations = 50

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPOAgent(
    state_dim, action_dim, epochs, horizon_len, max_epoch_len, actor_learning_rate, critic_learning_rate,
    train_actor_iterations, train_critic_iterations, gamma, lambda_, epsilon)
print(agent.actor)
print(agent.critic)

agent.run(env)
