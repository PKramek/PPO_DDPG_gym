import configparser

# This file is only used to create config.ini file
config = configparser.ConfigParser()

######################## PPO #######################
horizon_len = 4000
epochs = 250
gamma = 0.99
epsilon = 0.2
lambda_ = 0.95
actor_learning_rate = 1e-3
critic_learning_rate = 1e-3
train_actor_iterations = 80
train_critic_iterations = 80
minibatch_size = 64
hidden_size = 64

config['PPO'] = {
    'horizon_length': horizon_len,
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

config['PPO_DEFAULT'] = {
    'horizon_length': horizon_len,
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

######################## DDPG #######################
epochs_num = 2500
horizon_len = 1e6
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

config['DDPG'] = {
    'epochs_num': epochs_num,
    'horizon_len': horizon_len,
    'timesteps_per_epoch': timesteps_per_epoch,
    'actor_lr': actor_lr,
    'critic_lr': critic_lr,
    'start_steps': start_steps,
    'action_noise': action_noise,
    'max_ep_length': max_ep_length,
    'update_after': update_after,
    'update_every': update_every,
    'update_times': update_times,
    'batch_size': batch_size,
    'gamma': gamma,
    'polyak': polyak
}
config['DDPG_DEFAULT'] = {
    'epochs_num': epochs_num,
    'horizon_len': horizon_len,
    'timesteps_per_epoch': timesteps_per_epoch,
    'actor_lr': actor_lr,
    'critic_lr': critic_lr,
    'start_steps': start_steps,
    'action_noise': action_noise,
    'max_ep_length': max_ep_length,
    'update_after': update_after,
    'update_every': update_every,
    'update_times': update_times,
    'batch_size': batch_size,
    'gamma': gamma,
    'polyak': polyak
}

with open('config.ini', 'w') as configfile:
    config.write(configfile)
