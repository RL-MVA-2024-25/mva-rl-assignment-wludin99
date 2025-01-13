import torch
import torch.nn as nn

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = torch.nn.functional.leaky_relu(self.fc3(x))
        return self.fc4(x)

class DuelingQNetwork(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(DuelingQNetwork, self).__init__()
        self.advantage_net = QNetwork(n_input, n_hidden, n_output)
        self.value_net = QNetwork(n_input, n_hidden, 1)

    def forward(self, x):
        advantage = self.advantage_net(x)
        value = self.value_net(x)
        return value + advantage - advantage.mean()