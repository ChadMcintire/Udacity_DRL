import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    
    #class setup
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        #https://pytorch.org/docs/stable/nn.html
        
        #torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
        #bias – If set to False, the layer will not learn an additive bias. Default: True
        "*** YOUR CODE HERE ***"
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc1_units, fc2_units)
        self.fc4 = nn.Linear(fc2_units, action_size)
        
    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        #feed each layer it's intitial value
        #https://pytorch.org/docs/stable/nn.functional.html
        #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
