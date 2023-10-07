
import torch.nn as nn


import torch.nn as nn

class MLP_CelebA(nn.Module):
    """Used for celeba experiment"""

    def __init__(self, input_dim, num_classes):
        super(MLP_CelebA, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 500)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(500, 100)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(100, num_classes)  # Output layer with num_classes neurons

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x



class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x