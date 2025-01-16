import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GLSTM4(nn.Module):
    """
    Less GCN Layers
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, output_dim, dropout, activation_function):
        super(GLSTM4, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.lstm1 = nn.LSTM(input_size=hidden_dim1, hidden_size=hidden_dim1, num_layers=1, batch_first=True)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, hidden_dim3)
        self.conv4 = GCNConv(hidden_dim3, hidden_dim4)
        self.conv5 = GCNConv(hidden_dim4, hidden_dim5)
        self.fc = nn.Linear(hidden_dim5, output_dim)
        self.dropout = dropout
        self.activation_function = activation_function

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = x.unsqueeze(0)
        x, _ = self.lstm1(x)
        x = x.squeeze(0)
        
        x = self.conv2(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv4(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv5(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        #x = self.conv6(x, edge_index)

        #x = x.unsqueeze(0)
        #x, _ = self.lstm2(x)
        #x = x.squeeze(0)

        x = self.fc(x)
        
        return torch.sigmoid(x)



class GLSTM7(nn.Module):
    """
    Model that allows hyperparameter tuning on GLSTM4.
    """
    def __init__(self, input_dim, num_lstm_layers, lstm_hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, output_dim, dropout):
        super(GLSTM7, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim

        # Define GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, hidden_dim3)
        self.conv4 = GCNConv(hidden_dim3, hidden_dim4)
        self.conv5 = GCNConv(hidden_dim4, hidden_dim5)

        # Define LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=hidden_dim5 if i == 0 else lstm_hidden_dim,
                    hidden_size=lstm_hidden_dim,
                    num_layers=1,
                    batch_first=True)
            for i in range(num_lstm_layers)
        ])

        # Define fully connected layers
        self.fc = nn.Linear(hidden_dim5, output_dim)
        self.dropout = dropout
        self.activation_function = torch.relu

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)

        x = self.conv2(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv4(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv5(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc(x)
        return torch.sigmoid(x)





