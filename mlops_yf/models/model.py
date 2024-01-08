import torch

from torch import nn
import torch.nn.functional as F
class MyNeuralNet(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784,256)   # 784 -> 256
        self.fc2 = nn.Linear(256,128)   # 256 -> 128
        self.fc3 = nn.Linear(128,64)    # 128 -> 64
        self.fc4 = nn.Linear(64,10)     # 64 -> 10
        self.dropout = nn.Dropout(p=0.2)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        if x.ndim != 2:
            raise ValueError("Expected input tensor to have 2 dimensions, "
                             "got tensor with shape {x.shape}")
        if x.shape[1] != 784:
            raise ValueError("Expected input tensor to have 784 features, "
                             "got tensor with shape {x.shape}")
        x = x.view(x.shape[0],-1)   # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x),dim=1)    # output
        return x