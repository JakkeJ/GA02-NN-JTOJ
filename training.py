'''
script for training the agent for snake using various methods
'''

from tqdm import tqdm
import pandas as pd
import time
from utils import play_game2
from game_environment import SnakeNumpy
import torch
from DeepQAgent_pt import DeepQLearningAgent
import json

# some global variables
torch.manual_seed(42)
version = 'v17.1'
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_built() else device

print(f'Running on device: {device}')

# get training configurations
with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
    buffer_size = m['buffer_size']

# define no of episodes, logging frequency
episodes = 200000
log_frequency = 500
games_eval = 12

# setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions, 
                           buffer_size=buffer_size, version=version, device=device)
#agent.print_models()

# check in the same order as class hierarchy
if(isinstance(agent, DeepQLearningAgent)):
    agent_type = 'DeepQLearningAgent'
print('Agent is {:s}'.format(agent_type))

# setup the epsilon range and decay rate for epsilon
# define rewrad type and update frequency, see utils for more details
epsilon, epsilon_end = 1, 0.01
reward_type = 'discounted_future'
sample_actions = False
n_games_training = 8*16
decay = 0.97
iteration = 1194000
if(supervised):
    # lower the epsilon since some starting policy has already been trained
    epsilon = 0.01
    # load the existing model from a supervised method
    # or some other pretrained model
    agent.load_model(file_path='models/{:s}'.format(version), iteration = iteration)

# play some games initially to fill the buffer
# or load from an existing buffer (supervised)
if(supervised):
    try:
        agent.load_buffer(file_path='models/{:s}'.format(version), iteration = iteration)
        print("Buffer file found!")
    except FileNotFoundError:
        print("Buffer file not found!")
else:
    iteration = 0
    # setup the environment
    games = 8*16
    env = SnakeNumpy(board_size=board_size, frames=frames, 
                max_time_limit=max_time_limit, games=games,
                frame_mode=True, obstacles=obstacles, version=version)
    ct = time.time()
    _ = play_game2(env, agent, n_actions, n_games=games, record=True,
                    epsilon=epsilon, verbose=True, reset_seed=False,
                    frame_mode=True, total_frames=games*64)
    print('Playing {:d} frames took {:.2f}s'.format(games*64, time.time()-ct))

env = SnakeNumpy(board_size=board_size, frames=frames, 
            max_time_limit=max_time_limit, games=n_games_training,
            frame_mode=True, obstacles=obstacles, version=version)
env2 = SnakeNumpy(board_size=board_size, frames=frames, 
            max_time_limit=max_time_limit, games=games_eval,
            frame_mode=True, obstacles=obstacles, version=version)

# training loop
model_logs = {'iteration':[], 'reward_mean':[],
              'length_mean':[], 'games':[], 'loss':[]}
for index in tqdm(range(episodes)):
    # make small changes to the buffer and slowly train
    _, length, _ = play_game2(env, agent, n_actions, epsilon=epsilon,
                    n_games=n_games_training, record=True,
                    sample_actions=sample_actions, reward_type=reward_type,
                    frame_mode=True, total_frames=n_games_training, 
                    stateful=True)
    loss = agent.train_agent(batch_size=64, num_games=n_games_training, reward_clip=True, length = length)

    # check performance every once in a while
    if((index+1)%log_frequency == 0):
        # keep track of agent rewards_history
        current_rewards, current_lengths, current_games = \
                    play_game2(env2, agent, n_actions, n_games=games_eval, epsilon=-1,
                               record=False, sample_actions=False, frame_mode=True, 
                               total_frames=-1, total_games=games_eval)
        
        model_logs['iteration'].append(index + 1 + iteration)
        model_logs['reward_mean'].append(round(int(current_rewards)/current_games, 2))
        # model_logs['reward_dev'].append(round(np.std(current_rewards), 2))
        model_logs['length_mean'].append(round(int(current_lengths)/current_games, 2))
        model_logs['games'].append(current_games)
        model_logs['loss'].append(loss)
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'length_mean', 'games', 'loss']]\
          .to_csv('model_logs/{:s}.csv'.format(version), index=False)

    # copy weights to target network and save models
    if((index+1)%log_frequency == 0):
        agent.update_target_net()
        agent.save_model(file_path='models/{:s}'.format(version), iteration=(index + 1 + iteration))
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
