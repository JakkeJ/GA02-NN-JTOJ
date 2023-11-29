from collections import deque
import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
import json
import torch.optim as optim
from torchsummary import summary
from agent_pt import Agent

class DQN(nn.Module):
    def __init__(self, input_size, n_actions, version):
        super(DQN, self).__init__()
        self.input_size = input_size

        with open(f'model_config/{version}.json', 'r') as f:
            config = json.load(f)

        self.board_size = config['board_size']

        #Dynamic convolutional layers based on the config JSON-file in model_config folder
        #This so I can use the same script to run multiple sizes of models without changing the code
        self.conv_layers = nn.ModuleList()
        current_channels = input_size 
        for layer_name, layer_params in config['model'].items():
            if 'Conv2D' in layer_name:
                conv_layer = nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=layer_params['filters'],
                    kernel_size=tuple(layer_params['kernel_size']),
                    stride=layer_params.get('stride', 1),
                    padding=layer_params.get('padding', 0)
                )
                self.conv_layers.append(conv_layer)
                current_channels = layer_params['filters']

        #To use the model, we need to know the output size of the convolutional layers.
        #We can calculate this by passing a dummy input through the layers and checking the output size.
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_size, self.board_size, self.board_size)
            x = dummy_input
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            conv_output_size = x.view(x.size(0), -1).shape[1]

        #Dynamic linear layers based on the config JSON-file in model_config folder
        #This so I can use the same script to run multiple sizes of models without changing the code
        self.linear_layers = nn.ModuleList()
        linear_input_size = conv_output_size
        for layer_name, layer_params in config['model'].items():
            if 'Dense' in layer_name:
                linear_layer = nn.Linear(
                    in_features=linear_input_size,
                    out_features=layer_params['units']
                )
                self.linear_layers.append(linear_layer)
                linear_input_size = layer_params['units']

        self.output_layer = nn.Linear(linear_input_size, n_actions)

    #Dynamic forward pass based on the config JSON-file in model_config folder
    def forward(self, x):
        for layer in self.conv_layers:
            x = F.relu(layer(x))
        x = nn.Flatten()(x)
        
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        
        x = self.output_layer(x)
        return x
   
#For reference if a model doesn't work, this is the old hardcoded model.
#If need be, you can run the model through the model_converter.py to get the new dynamic style model.
'''
Hardcoded old model, using a dynamic model now

class DQN(nn.Module):
    def __init__(self, input_size, n_actions, version):
        super(DQN, self).__init__()
        self.input_size = input_size
        # Load the model configuration from the JSON file
        with open('model_config/{:s}.json'.format(version), 'r') as f:
            m = json.loads(f.read())
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels = self.input_size,
                                       out_channels = m['model']['Conv2D']['filters'],
                                       kernel_size = tuple(m['model']['Conv2D']['kernel_size']),
                                       stride=1,
                                       padding = 1)
        self.conv2 = nn.Conv2d(in_channels = m['model']['Conv2D']['filters'],
                                       out_channels = m['model']['Conv2D_1']['filters'],
                                       kernel_size = tuple(m['model']['Conv2D_1']['kernel_size']),
                                       stride = 1,
                                       padding = 1)
        self.conv3 = nn.Conv2d(in_channels = m['model']['Conv2D_1']['filters'],
                                       out_channels = m['model']['Conv2D_2']['filters'],
                                       kernel_size = tuple(m['model']['Conv2D_2']['kernel_size']),
                                       stride = 1,
                                       padding = 1)
        self.conv4 = nn.Conv2d(in_channels = m['model']['Conv2D_2']['filters'],
                                       out_channels = m['model']['Conv2D_3']['filters'],
                                       kernel_size = tuple(m['model']['Conv2D_3']['kernel_size']),
                                       stride = 1,
                                       padding = 1)
        self.conv5 = nn.Conv2d(in_channels = m['model']['Conv2D_3']['filters'],
                                       out_channels = m['model']['Conv2D_4']['filters'],
                                       kernel_size = tuple(m['model']['Conv2D_4']['kernel_size']),
                                       stride = 1)
        # Define the fully connected layers
        self.fc1 = nn.Linear(1024, m['model']['Dense_1']['units'])
        self.fc2 = nn.Linear(m['model']['Dense_1']['units'], m['model']['Dense_2']['units'])
        self.fc3 = nn.Linear(m['model']['Dense_2']['units'], n_actions)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
'''

class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values

    Attributes
    ----------
    _model : PyTorch Module
        Stores the PyTorch module of the DQN model
    _target_net : PyTorch Module
        Stores the target network module of the DQN model
    """
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version='', device="cpu"):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        self.device = device
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        self.reset_models()

    def reset_models(self):
        """Reset all the models by creating new instances"""
        self._model, self._loss_fn, self._optimizer = self._agent_model()
        self._model = self._model.to(self.device)
        if(self._use_target_net):
            self._target_net, _, _ = self._agent_model()
            self.update_target_net()

    def _prepare_input(self, board):
        """Reshape input and normalize"""
        #Be certain it is a PyTorch Tensor
        if isinstance(board, np.ndarray):
            board = torch.tensor(board, dtype=torch.float32)

        if board.ndim == 3:
        #Unsqueeze and permute to match PyTorch's format (N, C, H, W)
        #This is only for a single board
            board = board.permute(2, 0, 1).unsqueeze(0)
        else:
        #Permute the dimensions to match PyTorch's format (N, C, H, W)
            board = board.permute(0, 3, 1, 2)

        #Clone the tensor, normalize and return
        board = self._normalize_board(board.clone())
        return board.to(self.device)

    def _get_model_outputs(self, board, model=None):
        """Get action values from the DQN model"""
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model
        # forward pass through the model
        model_outputs = model(board).to(self.device)

        return model_outputs

    def _normalize_board(self, board):
        """Normalize the board before input to the network"""
        board = board / 4.0

        return board

    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value"""
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._model).to(self.device)
        legal_moves_tensor = torch.tensor(legal_moves).to(self.device)

        return torch.argmax(torch.where(legal_moves_tensor == 1, model_outputs, torch.full_like(model_outputs, -np.inf)), axis=1).cpu().numpy()

    def _agent_model(self):
        """Returns the model which evaluates Q values for a given state input"""
        n_frames = self._n_frames
        n_actions = self._n_actions
        model = DQN(n_frames, n_actions, self._version)

        loss_fn = nn.HuberLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0005)

        return model, loss_fn, optimizer

    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes
        
        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1).reshape((-1,1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        """Save the current models to disk using PyTorch's
        inbuilt save function.
        
        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        torch.save(self._model.state_dict(), "{}/model_{:04d}.pt".format(file_path, iteration))
        if(self._use_target_net):
            torch.save(self._target_net.state_dict(), "{}/model_{:04d}_target.pt".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        """Load models from disk using PyTorch's inbuilt load function
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        try:
            model_file_path = "{}/model_{:04d}.pt".format(file_path, iteration)
            target_model_file_path = "{}/model_{:04d}_target.pt".format(file_path, iteration)

            print("Attempting to load model from:", model_file_path)
            self._model.load_state_dict(torch.load(model_file_path))

            if self._use_target_net:
                print("Attempting to load target model from:", target_model_file_path)
                self._target_net.load_state_dict(torch.load(target_model_file_path))
        except FileNotFoundError:
            print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        """Print the current models using summary method"""
        print('Training Model')
        print(summary(self._model, (2, 10, 10)))
        if(self._use_target_net):
            print('Target Network')
            print(summary(self._target_net.to(self.device), (2, 10, 10)))

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False, length=1):
        """Train the model by sampling from buffer and return the error."""
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if(reward_clip):
            r = np.sign(r)

        #Arrays for training, converted to PyTorch tensors
        s = torch.tensor(s, dtype = torch.float32).to(self.device)
        a = torch.tensor(a, dtype = torch.float32).to(self.device)
        r = torch.tensor(r, dtype = torch.float32).to(self.device)
        next_s = torch.tensor(next_s, dtype = torch.float32).to(self.device)
        done = torch.tensor(done, dtype = torch.float32).to(self.device)
        legal_moves = torch.tensor(legal_moves, dtype = torch.float32).to(self.device)

        self._model.train()

        # calculate the discounted reward, and then train accordingly
        current_model = self._target_net if self._use_target_net else self._model
        current_model = current_model.to(self.device)
        with torch.no_grad():
            next_model_outputs = self._get_model_outputs(next_s, current_model)

        # our estimate of expected future discounted reward
        infinite_tensor = torch.tensor(-np.inf).to(self.device)
        discounted_reward = r + self._gamma * torch.max(torch.where(legal_moves == 1,
                                                                    next_model_outputs,
                                                                    infinite_tensor), dim = 1).values.reshape(-1, 1) * (1 - done)
        # create the target variable, only the column with action has different value
        target = self._get_model_outputs(s, current_model)
        # we bother only with the difference in reward estimate at the selected action
        target = (1 - a) * target + a * discounted_reward

        self._model.zero_grad()  # reset gradients

        output = self._get_model_outputs(s)  # forward pass
        loss = self._loss_fn(output, target)  # calculate loss
        loss.backward()  # backward pass
        self._optimizer.step()  # update weights
        return loss.item()

    def update_target_net(self):
        """Update the weights of the target network."""
        if(self._use_target_net):
            self._target_net.load_state_dict(self._model.state_dict())

    def compare_weights(self):
        """Check if the model and target network have the same weights or not."""
        for (name1, param1), (name2, param2) in zip(self._model.named_parameters(), self._target_net.named_parameters()):
            print('Layer {} Weights Match : {}'.format(name1, torch.equal(param1, param2)))

    def copy_weights_from_agent(self, agent_for_copy):
        """Update weights between competing agents which can be used in parallel training"""
        assert isinstance(agent_for_copy, self.__class__), "Agent type is required for copy"

        self._model.load_state_dict(agent_for_copy._model.state_dict())
        self._target_net.load_state_dict(agent_for_copy._target_net.state_dict())


class SupervisedLearningAgent(DeepQLearningAgent):
    """This agent learns in a supervised manner. A close to perfect
    agent is first used to generate training data, playing only for
    a few frames at a time, and then the actions taken by the perfect agent
    are used as targets. This helps learning of feature representation
    and can speed up training of DQN agent later.
 
    Attributes
    ----------
    _model_action_out : TensorFlow Softmax layer
        A softmax layer on top of the DQN model to train as a classification
        problem (instead of regression)
    _model_action : TensorFlow Model
        The model that will be trained and is simply DQN model + softmax
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Initializer for SupervisedLearningAgent, similar to DeepQLearningAgent
        but creates extra layer and model for classification training
        """        
        DeepQLearningAgent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        # define model with softmax activation, and use action as target
        # instead of the reward value
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self._model.parameters(), lr=0.0005)
 
        
    def train_agent(self, batch_size=32, num_games=1, epochs=5, reward_clip=False):
        self._model.train()  #Set the model to training mode
 
        last_epoch_loss = 0
        for epoch in range(epochs):
            total_loss = 0
            for _ in range(num_games):
                #Sample data from the buffer
                s, a, _, _, _, _ = self._buffer.sample(batch_size)
 
                # Convert data to PyTorch tensors
                states = torch.tensor(s, dtype=torch.float32)
                states = states.permute(0, 3, 1, 2)
                actions = torch.tensor(a, dtype=torch.float32)
 
                #Forward pass: Compute predicted y by passing states to the model
                outputs = self._model(states)
 
                #Compute loss
                loss = self.loss_function(outputs, actions)
 
                #Zero gradients, backward pass, optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
 
                total_loss += loss.item()
 
            #Average loss for the current epoch
            avg_loss = total_loss / num_games
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}')
 
            #Store the average loss of the last epoch
            if epoch == epochs - 1:
                last_epoch_loss = avg_loss
 
        #Return the average loss of the last epoch
        return last_epoch_loss
 
    def get_max_output(self):
        self._model.eval()
 
        s, _, _, _, _, _ = self._buffer.sample(self.get_buffer_size())
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        s = s.permute(0, 3, 1, 2)
        normalized_s = self._normalize_board(s)
 
        with torch.no_grad():
            outputs = self._model(normalized_s)
 
        max_value = torch.max(torch.abs(outputs)).item()
 
        return max_value
 
    def normalize_layers(self, max_value=None):
        #Check if max_value is valid, if not set to 1.0
        if max_value is None or math.isnan(max_value):
            max_value = 1.0
 
        #Normalize the weights of the last fully connected layer
        with torch.no_grad():
            self._model.output_layer.weight /= max_value
            self._model.output_layer.bias /= max_value
            
class BreadthFirstSearchAgent(Agent):
    """
    finds the shortest path from head to food
    while avoiding the borders and body
    """
    def _get_neighbors(self, point, values, board):
        """
        point is a single integer such that 
        row = point//self._board_size
        col = point%self._board_size
        """
        row, col = self._point_to_row_col(point)
        neighbors = []
        for delta_row, delta_col in [[-1,0], [1,0], [0,1], [0,-1]]:
            new_row, new_col = row + delta_row, col + delta_col
            if(board[new_row][new_col] in \
               [values['board'], values['food'], values['head']]):
                neighbors.append(new_row*self._board_size + new_col)
        return neighbors
 
    def _get_shortest_path(self, board, values):
        # get the head coordinate
        board = board[:,:,0]
        head = ((self._board_grid * (board == values['head'])).sum())
        points_to_search = deque()
        points_to_search.append(head)
        path = []
        row, col = self._point_to_row_col(head)
        distances = np.ones((self._board_size, self._board_size)) * np.inf
        distances[row][col] = 0
        visited = np.zeros((self._board_size, self._board_size))
        visited[row][col] = 1
        found = False
        while(not found):
            if(len(points_to_search) == 0):
                # complete board has been explored without finding path
                # take any arbitrary action
                path = []
                break
            else:
                curr_point = points_to_search.popleft()
                curr_row, curr_col = self._point_to_row_col(curr_point)
                n = self._get_neighbors(curr_point, values, board)
                if(len(n) == 0):
                    # no neighbors available, explore other paths
                    continue
                # iterate over neighbors and calculate distances
                for p in n:
                    row, col = self._point_to_row_col(p)
                    if(distances[row][col] > 1 + distances[curr_row][curr_col]):
                        # update shortest distance
                        distances[row][col] = 1 + distances[curr_row][curr_col]
                    if(board[row][col] == values['food']):
                        # reached food, break
                        found = True
                        break
                    if(visited[row][col] == 0):
                        visited[curr_row][curr_col] = 1
                        points_to_search.append(p)
        # create the path going backwards from the food
        curr_point = ((self._board_grid * (board == values['food'])).sum())
        path.append(curr_point)
        while(1):
            curr_row, curr_col = self._point_to_row_col(curr_point)
            if(distances[curr_row][curr_col] == np.inf):
                # path is not possible
                return []
            if(distances[curr_row][curr_col] == 0):
                # path is complete
                break
            n = self._get_neighbors(curr_point, values, board)
            for p in n:
                row, col = self._point_to_row_col(p)
                if(distances[row][col] != np.inf and \
                   distances[row][col] == distances[curr_row][curr_col] - 1):
                    path.append(p)
                    curr_point = p
                    break
        return path
 
    def move(self, board, legal_moves, values):
        if(board.ndim == 3):
            board = board.reshape((1,) + board.shape)
        board_main = board.copy()
        a = np.zeros((board.shape[0],), dtype=np.uint8)
        for i in range(board.shape[0]):
            board = board_main[i,:,:,:]
            path = self._get_shortest_path(board, values)
            if(len(path) == 0):
                a[i] = 1
                continue
            next_head = path[-2]
            curr_head = (self._board_grid * (board[:,:,0] == values['head'])).sum()
            # get prev head position
            if(((board[:,:,0] == values['head']) + (board[:,:,0] == values['snake']) \
                == (board[:,:,1] == values['head']) + (board[:,:,1] == values['snake'])).all()):
                # we are at the first frame, snake position is unchanged
                prev_head = curr_head - 1
            else:
                # we are moving
                prev_head = (self._board_grid * (board[:,:,1] == values['head'])).sum()
            curr_head_row, curr_head_col = self._point_to_row_col(curr_head)
            prev_head_row, prev_head_col = self._point_to_row_col(prev_head)
            next_head_row, next_head_col = self._point_to_row_col(next_head)
            dx, dy = next_head_col - curr_head_col, -next_head_row + curr_head_row
            if(dx == 1 and dy == 0):
                a[i] = 0
            elif(dx == 0 and dy == 1):
                a[i] = 1
            elif(dx == -1 and dy == 0):
                a[i] = 2
            elif(dx == 0 and dy == -1):
                a[i] = 3
            else:
                a[i] = 0
        return a
 
    def get_action_proba(self, board, values):
        """ for compatibility """
        move = self.move(board, values)
        prob = [0] * self._n_actions
        prob[move] = 1
        return prob
 
    def _get_model_outputs(self, board=None, model=None):
        """ for compatibility """ 
        return [[0] * self._n_actions]
 
    def load_model(self, **kwargs):
        """ for compatibility """
        pass