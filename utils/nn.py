import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_sizes, output_size, pos_weight=None, lr=0.01, device='cpu', hidden_activation='relu', output_activation='sigmoid', criterion='NLLLoss'):
        super(NeuralNetwork, self).__init__()

        # Dictionary for activation functions
        self.activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(dim=1)}
        
        # Setting criterion
        if criterion == 'BCELoss':
            self.criterion = nn.BCELoss()
        elif criterion == 'BCEWithLogitsLoss':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            print('Default BCELoss criterion')
            self.criterion = nn.BCELoss()
        
        # Creating the hidden layers
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(self.activations[hidden_activation])
            current_input_size = hidden_size
        
        # Adding the drop out and output layer
        layers.append(nn.Dropout())
        layers.append(nn.Linear(current_input_size, output_size))
        layers.append(self.activations[output_activation])
                  
        # Creating a Sequential model
        self.fc_layers = nn.Sequential(*layers)
        
        self.lr = lr
        self.device = device
        
    def forward(self, x):
        return self.fc_layers(x) #retorna sigmoid

    def predict(self, x, threshold):
        y_hat = self.forward(x)
        if threshold:
            y_pred = torch.where(y_hat > threshold, 1.0, 0.0)
        else:
            y_pred = torch.round(y_hat)
        y_pred = y_pred.detach().numpy()
        return y_pred
