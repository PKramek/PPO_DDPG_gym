# PPO and DDPG for gym

## CLI
CLI was implemented to allow users to easily run implemented algorithms.
Available commmands:

- --help - displays helpfull information about other commands,
- -a --algortihm - allows user to chose which algorithm to choose. Only PPO and DDPG are available,
- -c --config - which configuration to use for algorithm. Name of the secrtion from config.ini file should be given. Available options are PPO, DDPG, PPO_DEFAULT, DDPG_DEFAULT. If none is given, section with the same name as algorithm will be used.
- -e --environment - which gym environment to use. Additional evironment DoublePrecisionSwimmer-v2 is available, which is Swimmer-v2 environment with double time discretization,
- -np --no-plot - if this flag is raised after the training process plot showing average episode return during training wont be created,
- -p --play - if this flag is raised agents will be used in play mode, which allows You to see achived results. When this flag is raised You must provide paths to files containg trained neural networks,
- -ap --actor-path - path to file containing actor neural network,
- -cp --critic-path - path to file containing critic neural network
- -acp --actor-critic-path - path to file contaning both critic and actor neural networks

## Running

To for example run training for PPO agent using Swimmer-v2 envorionment use command:

python main.py -a PPO -e Swimmer-v2

To run PPO in play mode in Swimmer-v2 environment using trained model use:

python main.py -a PPO -e Swimmer-v2 -p -ap ./trained_models/PPO/best/Swimmer-v2_actor_250_epochs.pkl -cp ./trained_models/PPO/best/Swimmer-v2_critic_250_epochs.pkl
