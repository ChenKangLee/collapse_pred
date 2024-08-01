import torch
import torch.nn as nn


# it seems like FCU version is not really using GCN
class FCU(nn.Module):
    def __init__(self, dim_rain, dim_geo, device=torch.device('cpu'), dropout_rate=0.4):
        super(FCU, self).__init__()
        
        self.dim_rain = dim_rain
        self.dim_geo = dim_geo

        self.device = device
        self.dropout_rate = dropout_rate

        self._build_net()


    def _build_net(self):
        self.geo_fc = nn.Sequential(
            nn.Linear(self.dim_geo, self.dim_geo * 2),
            nn.BatchNorm1d(self.dim_geo * 2),
            nn.ReLU(),
        )

        self.lstm1 = nn.LSTM(self.dim_rain, self.dim_rain * 2, batch_first=True)

        # the input will be the concatenated output of the `geo_fc` and `rain_lstm` layers
        self.fc = nn.Sequential(
            nn.Linear(self.dim_geo * 2 + self.dim_rain * 2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )


    def forward(self, rain, geo):
        rain = rain.float()
        geo = geo.float()

        # shape: (batch, dim_geo * 16)
        geo_emb = self.geo_fc(geo) # geo is in double for some reason, cast here
        
        # we are only using the output of the final iteration
        # shape: (batch, 1, dim_rain * 2) -> (batch, dim_rain * 2)
        lstm_out, _ = self.lstm1(rain)
        rain_emb = lstm_out[:, -1, :].reshape((-1, self.dim_rain * 2))

        emb_cat = torch.cat([geo_emb, rain_emb], dim=1)
        logits = self.fc(emb_cat)
        return logits


