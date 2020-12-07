import gym

gym.envs.register(
    id='DoublePrecisionSwimmer-v2',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    max_episode_steps=2000
)

env = gym.make('DoublePrecisionSwimmer-v2')
print("Action space: {}".format(env.action_space))
print("Observation space: {}".format(env.action_space))


def cut_timestep_in_half(env):
    env.model.opt.timestep = env.model.opt.timestep / 2


print(env.model.opt.timestep)
cut_timestep_in_half(env)
print(env.model.opt.timestep)

for i_episode in range(100):
    observation = env.reset()
    for t in range(2000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action, )
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
