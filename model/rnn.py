import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid

class RNN(nn.Module):
    def __init__(self, dim_rain, dim_geo, dim_hidden, device=torch.device('cpu'), dropout_rate=0.4):
        super(RNN, self).__init__()

        self.dim_rain = dim_rain
        self.dim_geo = dim_geo
        self.dim_hidden = dim_hidden
        self.device = device
        self.dropout_rate = dropout_rate

        self._build_net()

    
    def _build_net(self):
        # self.embed = nn.Sequential(
        #     nn.Linear(self.dim_geo, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, self.dim_hidden)
        # )

        # self.lstm = nn.LSTM(input_size=2, hidden_size=self.dim_hidden, num_layers=1, batch_first=True)
        self.lstm = nn.LSTM(input_size=self.dim_rain + self.dim_geo, hidden_size=self.dim_hidden, num_layers=4, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(self.dim_hidden * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, rain, geodata):
        # generate initial hidden weights with the `embed` layers
        # hidden = self.embed(geodata)

        geodata = torch.unsqueeze(geodata, 1).repeat(1, rain.shape[1], 1)
        lstm_in = torch.cat([rain, geodata], dim=2)

        lstm_in = self.dropout(lstm_in)
        
        # hidden = torch.unsqueeze(hidden, 0) # for LSTM input
        # cell_0 = torch.rand((1, hidden.shape[1], self.dim_hidden)).to(self.device)
        # _, (hidden_out, _) = self.lstm(rain, (hidden, cell_0))

        output, (hidden_out, cell) = self.lstm(lstm_in)

        # use final hidden layer to predict collpase
        flattened = torch.cat([hidden_out[0], hidden_out[1], hidden_out[2], hidden_out[3]], dim=1)

        flattened = self.dropout(flattened)
        out = self.fc(flattened)

        return out.view(rain.shape[0], 1) # remove the dim 1 in axis 1, resulting from LSTM input format
