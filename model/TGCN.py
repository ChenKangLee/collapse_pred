import torch
import torch.nn as nn
from model.graph import GraphConvLayer
from model.GCNGRU import GCNGRU


class TGCN(nn.Module):
    def __init__(self, dim_rain, dim_geo, n_slopeunits, laplacian, device=torch.device('cpu'), dropout_rate=0.4):
        super(TGCN, self).__init__()

        self.dim_rain = dim_rain
        self.dim_geo = dim_geo
        self.n_slopeunits = n_slopeunits

        # hard-code hyperparam for now
        self.GRU_IN = self.dim_rain + self.dim_geo
        self.GRU_HIDDEN = self.dim_rain + self.dim_geo

        self.device = device
        self.dropout_rate = dropout_rate

        self._build_net(laplacian)

    
    def _build_net(self, laplacian):
        self.gcngru = GCNGRU(self.GRU_IN, laplacian)
        self.fc = nn.Sequential(
            nn.Linear(self.GRU_HIDDEN, 8),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1)
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

        # `gruout` shape: (B, N, GRU_HIDDEN)
        gru_out = self.gcngru(inputs)

        # `outputs` shape: (B, 1)
        # TODO: maybe will cause problems?
        gru_out = gru_out.reshape((B*N, self.GRU_HIDDEN))
        outputs = self.fc(gru_out)

        # bring back dimension of slope units
        outputs = outputs.reshape((B, N, -1))
        return outputs

