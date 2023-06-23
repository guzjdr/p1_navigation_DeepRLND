from common import *

class DeepQNetwork(nn.Module):
    """ Building a Deep Q Network with Pytorch """
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        Neural net model parameters
        Params:
        ******
            state_size (int): Observation (state) vector size
            action_size (int): Action vector size
            seed (int): Random seed
            fc1_units (int): Number of nodes in first network layer
            fc1_unit (int): Number of nodes in second network layer
        """
        super(DeepQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        """
        Forward pass of the state through the model to compute the corresponding action. 
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)