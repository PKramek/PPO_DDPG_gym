import argparse

import gym
from gym import envs

from Agents.DDPG.ddpgagent import DDPGAgent
from Agents.PPO.ppoagent import PPOAgent

gym.envs.register(
    id='DoublePrecisionSwimmer-v2',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    max_episode_steps=2000
)


def cut_timestep_in_half(env):
    env.model.opt.timestep = env.model.opt.timestep / 2


def get_env_spec(environment):
    all_envs = envs.registry.all()
    environment = [env_spec for env_spec in all_envs if env_spec.id == environment]
    if len(environment) == 0:
        return None
    else:
        return environment[0]


parser = argparse.ArgumentParser(prog="PPO DDPG Gym solver",
                                 description='This program allows you to perform RL algorithms on Gym environments')

parser.add_argument('-a', '--algorithm', type=str, required=True,
                    help='Algorithm used to solve Gym environment',
                    choices=['PPO', 'DDPG'])

parser.add_argument('-c', '--config', type=str, required=False, default=None,
                    help='Which configuration should be used to initialize learning algorithm, if none is given version non-default version for a given algorithm will be used',
                    choices=['PPO', 'DDPG', 'PPO_DEFAULT', 'DDPG_DEFAULT'])

parser.add_argument('-e', '--environment', type=str, required=True,
                    help='What gym environment should be used, there is also additional environment provided: DoublePrecisionSwimmer-v2, which is modified version of Swimmer with double time precision')

parser.add_argument('-np', '--no-plot', action='store_true',
                    help='If this flag is passed average episode return will not be plotted in relation to number of timesteps, if agent is not trained this flag is ignored')

parser.add_argument('-pp', '--plot-path', type=str, default=None,
                    help='Path to file in which generated plot will be saved, if none is given plot will be saved in file /results/<Algorithm_Name>.png')

parser.add_argument('-p', '--play', action='store_true',
                    help='If this flag is passed, then agent will not be trained, but will use already trained models')

parser.add_argument('-ap', '--actor-path', type=str, default=None,
                    help='Path to pickle file containing trained actor model')
parser.add_argument('-cp', '--critic-path', type=str, default=None,
                    help='Path to pickle file containing trained critic model')
parser.add_argument('-acp', '--actor-critic-path', type=str, default=None,
                    help='Path to pickle file containing trained actor-critic model')

if __name__ == '__main__':

    algorithms_lookup = {
        'PPO': PPOAgent,
        'DDPG': DDPGAgent
    }
    args = parser.parse_args()

    algorithm = algorithms_lookup.get(args.algorithm, None)
    if algorithm is None:
        raise NotImplementedError(f'Algorithm {args.algorithm} not known')

    if args.config is None:
        # those names could be changed in the future
        config_section_lookup = {'PPO': 'PPO', 'DDPG': 'DDPG'}
        section = config_section_lookup.get(args.algorithm, None)
    else:
        section = args.config

    env_spec = get_env_spec(args.environment)

    if env_spec is None:
        raise ValueError(f"Unknown Gym environment: {args.environment}")
    else:
        env = gym.make(env_spec.id)

        if env_spec.id == 'DoublePrecisionSwimmer-v2':
            cut_timestep_in_half(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = env.action_space.high[0]

    agent = algorithm.from_config_file('config.ini', section, state_dim, action_dim, action_limit,
                                       env_spec.max_episode_steps)

    if args.play:
        if args.algorithm == 'PPO':
            assert args.actor_path is not None, 'If agent is used in play mode path to actor model must be provided'
            assert args.critic_path is not None, 'If agent is used in play mode path to critic model must be provided'

            agent.play(env, args.actor_path, args.critic_path)
        elif args.algorithm == 'DDPG':
            assert args.actor_critic_path is not None, 'If agent is used in play mode path to critic model must be provided'
            agent.play(env, args.actor_critic_path)
            agent.plot_episode_returns(f'./results/{env_spec.id}-{args.algorithm}.png')
        else:
            raise ValueError(f"Unknown Gym environment: {args.environment}")
    else:
        agent.train(env)

        if not args.no_plot:
            if args.plot_path is None:
                path = f'./results/{env_spec.id}-{args.algorithm}.png'
            else:
                path = args.plot_path
            agent.plot_episode_returns(path)
