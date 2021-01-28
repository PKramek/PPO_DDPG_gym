import configparser

# This file is only used to create config.ini file
config = configparser.ConfigParser()

######################## PPO #######################
horizon_len = 4000
epochs = 250
gamma = 0.995
double_precision_gamma = 0.999
epsilon = 0.2
lambda_ = 0.97
actor_learning_rate = 1e-3
critic_learning_rate = 1e-3
train_actor_iterations = 110
train_critic_iterations = 110
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

config['PPO_DoublePrecision'] = {
    'horizon_length': horizon_len,
    'epochs': epochs,
    'gamma': double_precision_gamma,
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
epochs = 250
double_precision_epochs = 125
horizon_length = 100000
episodes_in_epoch = 4
actor_learning_rate = 1e-3
critic_learning_rate = 1e-3
start_steps = 10000
action_noise = 0.1
update_after = 1000
double_precision_update_after = 2000
update_every = 50
batch_size = 100
gamma = 0.99
double_precision_gamma = 0.994
polyak = 0.995
hidden_size = 256


config['DDPG'] = {
    'epochs': epochs,
    'horizon_length': horizon_length,
    'episodes_in_epoch': episodes_in_epoch,
    'actor_learning_rate': actor_learning_rate,
    'critic_learning_rate': critic_learning_rate,
    'start_steps': start_steps,
    'action_noise': action_noise,
    'update_after': update_after,
    'update_every': update_every,
    'batch_size': batch_size,
    'gamma': gamma,
    'polyak': polyak,
    'hidden_size': hidden_size
}
config['DDPG_DoublePrecision'] = {
    'epochs': double_precision_epochs,
    'horizon_length': horizon_length,
    'episodes_in_epoch': episodes_in_epoch,
    'actor_learning_rate': actor_learning_rate,
    'critic_learning_rate': critic_learning_rate,
    'start_steps': start_steps,
    'action_noise': action_noise,
    'update_after': double_precision_update_after,
    'update_every': update_every,
    'batch_size': batch_size,
    'gamma': double_precision_gamma,
    'polyak': polyak,
    'hidden_size': hidden_size
}

with open('config.ini', 'w') as configfile:
    config.write(configfile)
