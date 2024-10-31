import torch.nn as nn

# Define the neural network model
class DNN_Model(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_layers, dropout):
        super(DNN_Model, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            if i < num_layers - 1:  # No ReLU/dropout after the last layer
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))
                
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x