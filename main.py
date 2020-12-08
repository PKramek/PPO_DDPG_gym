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


env = gym.make('DoublePrecisionSwimmer-v2')
print("Action space: {}".format(env.action_space.shape[0]))
print("Observation space: {}".format(env.observation_space.shape[0]))

horizon_len = 512
epochs = 50
steps_per_epoch = 4000
gamma = 0.99
epsilon = 0.2
actor_learning_rate = 3e-4
critic_learning_rate = 1e-3
train_actor_iterations = 80
train_critic_iterations = 80
lambda_ = 0.97

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPOAgent(
    state_dim, action_dim, epochs, horizon_len, actor_learning_rate, critic_learning_rate, gamma,
    lambda_, epsilon)
print(agent.actor)
print(agent.critic)
print(agent.device)

agent.run(env)
