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
train_actor_iterations = 150
train_critic_iterations = 150
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

config['DDPG'] = {}
config['DDPG_DEFAULT'] = {}

with open('config.ini', 'w') as configfile:
    config.write(configfile)
