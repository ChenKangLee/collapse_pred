import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid, Softmax
from model.graph import GraphConvLayer


class TGCN(nn.Module):
    def __init__(self, dim_rain, dim_geo, n_slopeunits, laplacian, device=torch.device('cpu'), dropout_rate=0.4):
        super(TGCN, self).__init__()

        self.dim_rain = dim_rain
        self.dim_geo = dim_geo
        self.n_slopeunits = n_slopeunits

        # hard code hyperparam for now
        self.DIM_GCN_EMB = 32
        self.LSTM_HIDDEN = self.n_slopeunits * self.DIM_GCN_EMB

        self.device = device
        self.dropout_rate = dropout_rate

        self._build_net(laplacian)

    
    def _build_net(self, laplacian):
        self.gcn = GraphConvLayer(self.dim_geo + self.dim_rain, self.DIM_GCN_EMB, laplacian)
        self.lstm = nn.LSTM(
            self.n_slopeunits * self.DIM_GCN_EMB,
            self.LSTM_HIDDEN,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.LSTM_HIDDEN, self.n_slopeunits * 2),
            nn.BatchNorm1d(self.n_slopeunits * 2),
            nn.ReLU(),
            nn.Linear(self.n_slopeunits * 2, self.n_slopeunits),
            nn.BatchNorm1d(self.n_slopeunits),
            nn.Sigmoid()
        )


    def forward(self, rain: torch.Tensor, geodata: torch.Tensor):
        # shape of input
        # B: batch size
        # N: number of slope units
        # T: sequence length
        
        # shape of `rain`: (B, N, T, 2)
        # shape of `geodata`: (B, N, 18)
        B, N, T, _ = rain.shape
        geodata = torch.unsqueeze(geodata, dim=2)
        # tile geodata for each timestep of rain input
        # (B, N, 16) -> (B, N, T, 18)
        geodata = torch.tile(geodata, (1, 1, T, 1))

        # concat geodata to the rain data at each timestamp of the sequence
        # inputs shape: (B, N, T, 20)
        inputs = torch.cat((rain, geodata), dim=3)

        # emb shape: (B, N, T, DIM_GCN_EMB)
        emb = self.gcn(inputs)

        # (B, N, T, DIM_GCN_EMB) -> (B, T, N, DIM_GCN_EMB)
        emb = emb.transpose(1, 2)

        # flatten all the values from each slope unit into one feature vector
        lstm_inputs = emb.reshape((B, T, N * self.DIM_GCN_EMB))

        # lstm_out shape: (B, T, DIM_LSTM_HIDDEN)
        lstm_out, (_, _) = self.lstm(lstm_inputs)

        # we are only taking the output at the end of the sequence
        # shape: (B, DIM_LSTM_HIDDEN)
        fc_inputs = lstm_out[:, -1, :]
        outputs = self.fc(fc_inputs)
        return outputs






