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
        self.DIM_GCN_EMB = 24
        self.LSTM_HIDDEN = 48

        self.device = device
        self.dropout_rate = dropout_rate

        self._build_net(laplacian)

    
    def _build_net(self, laplacian):
        self.gcn = GraphConvLayer(self.dim_geo + self.dim_rain, self.DIM_GCN_EMB, laplacian)
        self.lstm = nn.LSTM(
            self.DIM_GCN_EMB,
            self.LSTM_HIDDEN,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(self.LSTM_HIDDEN, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
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
        del inputs

        # flatten, remove the dimension of slopeunits
        # (B, N, T, DIM_GCN_EMB) -> (B * N, T, DIM_GCN_EMB)
        emb = emb.reshape((-1, T, self.DIM_GCN_EMB))

        # lstm_out shape: (B * N, T, DIM_LSTM_HIDDEN)
        lstm_out, (_, _) = self.lstm(emb)
        del emb

        # we are only taking the output at the end of the sequence
        # shape: (B * N, DIM_LSTM_HIDDEN)
        fc_inputs = lstm_out[:, -1, :]
        outputs = self.fc(fc_inputs)

        # reshape output to bring back dimension of slopeunits
        outputs = outputs.reshape((B, N, -1))
        return outputs






