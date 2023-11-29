# Snake Reinforcement Learning

Code for training a Deep Reinforcement Learning agent to play the game of Snake.
The agent takes 2 frames of the game as input (image) and predicts the action values for
the next action to take.

Converted from TensorFlow/Keras to PyTorch 2.1.0 by Jim Tore Jakobsen

## Running the code
Run the training.py, choose a pretrained model or run with a config file. There are some pretrained models and some buffers to test.

I have been using Python 3.11.6 and PyTorch 2.1.0 for this! The code might not run if you are not using at least PyTorch 2.1.0! Up to your discretion to try!

Best models are named like this:
best_8obs1_996000.pt and best_8obs1_996000_target.pt
Rename them to "model_XXXXXX.pt" and "model_XXXXXX_target.pt", these names are to show what stage of the run I was in. 8obs means 8 obstacles. The 1359500 and 996000 are similar in mean length, but one is trained more and on a lot more boards, so it should do better.

## Code Structure
[DeepQAgent_pt.py](../DeepQAgent_pt.py) contains the agent for playing the game. It implements and trains a convolutional neural network for the action values. Following classes are available
<table>
    <head>
        <tr>
        <th> Class </th><th> Description</th>
        </tr>
    </head>
    <tr><td>DeepQLearningAgent/td><td>Deep Q Learning Algorithm with CNN Network, PyTorch-based</td></tr>
    <tr><td>SupervisedLearningAgent</td><td>Trains Using Examples from another Agent/Human</td></tr>
    <tr><td>BreadthFirstSearchAgent</td><td>Repeatedly Finds Shortest Path from Snake Head to Food for Traversal</td></tr>
</table>

[agent_pt.py](../agent_pt.py) contains the base agent class.

[training.py](../training.py) contains the complete code to train an agent.

[game_visualization.py](../game_visualization.py) contains the code to convert the game to mp4 format.

[model_converter.py](../model_converter.py) contains code to convert older models using a fixed model to the new dynamic DQN model currently present in the code. For now, the old fixed model is present in the code, but commented out, for reference if there are any issues with running older models for visualization.

Changes from the original code by Jim: Added a limit so it doesn't render runs below a certain score, and if it has the same amount of food for a certain amount of frames. See below.

```python
from DeepQAgent_pt import DeepQLearningAgent
from game_environment import Snake
from utils import visualize_game
import json

# some global variables
version = 'v17.1'

with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])

iteration_list = [1965000]
max_time_limit = 998

# setup the environment
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit,
            obstacles=obstacles, version=version)
s = env.reset()
n_actions = env.get_num_actions()

# setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, 
                           n_actions=n_actions, buffer_size=10, version=version)

for iteration in iteration_list:
    agent.load_model(file_path='models/{:s}'.format(version), iteration=iteration)
    
    for i in range(50):
        visualize_game(env, agent,
            path='images/game_visual_{:s}_{:d}_14_ob_{:d}.mp4'.format(version, iteration, i),
            debug=False, animate=True, fps=12, minimum_score = 25, looping_length = 100)
```

## Experiments
Configuration for different experiments can be found in [/model_config/](../model_config/) files.
I have run 4 and 5 layer CNNs, where the 5 layer CNN had an extra Linear layer added, and with this, I saw runs with up to 46 in score on a board with 8 obstacles, which is just 8 squares away from a full board, since the snake starts at 2 in length.

To achieve this, I first trained it with a filled replay buffer, but no BFS model, 12 obstacles, than took the best model I had after 1 million episodes, made new boards with 8 obstacles, and continued training for 2 million episodes more. Note that I chose the model with the highest length when I retrained, I did not continue on the last model I trained (number 1 million), so the number of episodes does not match 1 million + 2 million here.

Due to what I've seen online from other PyTorch implementations, I expected less to be more (as in fewer layers), but from what I've seen with my experiments, more is more in this case. It starts climbing from ~2 in length a lot earlier compared to 3 and 4 layers CNN versions, and the peaks are a lot higher as well. The 3 layer one I trained for 2.4 million episodes without doing any changes did not give good results in testing, while the 1 million 5 layer one did.

My "official" submission for this GA02 task, is a 3 layer CNN as we are supposed to use, v17.1.json, with the pretrained BFS model and buffer, which I ran to 500k episodes on 36 boards with 12 obstacles, then I changed it to use the best model, 496500, and a new set of 360 boards with 8 obstacles. I have been running 12 games in my training.py, not 8 as it was in the TensorFlow version. I am not sure if there is any difference, but I have stuck to this through my experiments with 3 layer, 4 layer and 5 layer CNNs after the initial 2.4 million run. I ran 500k episodes with the new boards with 8 obstacles, then another 200k episodes with 12 obstacles with brand new boards, and then 200k with a new set of boards with 8 obstacles. It seems to learn do better when I alternate between different boards with different amounts of obstacles. I have gotten runs with up to 37 in score with this setup. It is not as good as the 5 layer CNN one without BFS (and I never got to test a 5 layer CNN with BFS to the same extent), but it is pretty good still!

### Change of Reward Type
I have changed the reward type from current to discounted_future, Q Learning is normally more towards using future rewards than current ones in general, so I took this decision.

### Batch Size
I have kept batch size at 64 as the original author claimed this was the best for his code.


### PreTraining
I have pretrained my "official" submission by rewriting the BFS and SupervisedLearningAgent to work with PyTorch, so I get a pretrained model_0000.pt to use alongside the buffer0001 file. As for the results of this, if I ran without any form of pretraining or buffer, I got a lot of looping. With just using the buffer, there was significantly less looping, so it seems to be necessary when using obstacles.

### Environment with Obstacles
There are samples in [/images/best](../images/best/)

