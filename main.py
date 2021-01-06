import gym
import torch

from PPO.ppoagent import PPOAgent
from DDPG.ddpgagent import DDPGAgent

assert torch.cuda.is_available()

gym.envs.register(
    id='DoublePrecisionSwimmer-v2',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    max_episode_steps=2000
)


def cut_timestep_in_half(env):
    env.model.opt.timestep = env.model.opt.timestep / 2


def get_ppo_agent():
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
    return agent


def get_ddpg_agent():
    epochs_num = 2500
    horizon_len = 4000
    timesteps_per_epoch = 4000
    actor_lr = 1e-3
    critic_lr = 1e-3
    start_steps = 10000
    action_noise = 0.1
    max_ep_length = 1000
    update_after = 1000
    update_every = 50
    update_times = 50
    batch_size = 100
    gamma = 0.99
    polyak = 0.995

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = env.action_space.high[0]

    ddpgagent = DDPGAgent(
        state_dim, action_dim, action_limit, epochs_num, horizon_len, timesteps_per_epoch, actor_lr, critic_lr,
        start_steps, action_noise, max_ep_length, update_after, update_every, update_times, batch_size, gamma, polyak)

    print(ddpgagent.actor_critic)
    return ddpgagent


env = gym.make('Swimmer-v2')

# agent = get_ppo_agent()
agent = get_ddpg_agent()

agent.run(env)
