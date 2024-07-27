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
            nn.Linear(self.dim_geo, self.dim_geo * 4),
            nn.BatchNorm1d(self.dim_geo * 4),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dim_geo * 4, self.dim_geo * 4 * 4),
            nn.BatchNorm1d(self.dim_geo * 4 * 4)
        )

        self.lstm1 = nn.LSTM(self.dim_rain, self.dim_rain * 4, batch_first=True)
        self.lstm2 = nn.LSTM(self.dim_rain * 4, self.dim_rain * 4 * 4, batch_first=True)

        # the input will be the concatenated output of the `geo_fc` and `rain_lstm` layers
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dim_geo * 4 * 4 + self.dim_rain * 4 * 4, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.Sigmoid(),
            nn.Linear(8, 1),
        )


    def forward(self, rain, geo):
        rain = rain.float()
        geo = geo.float()

        # shape: (batch, dim_geo * 16)
        geo_emb = self.geo_fc(geo) # geo is in double for some reason, cast here

        lstm_out_1, _ = self.lstm1(rain)
        
        # we are only using the output of the final iteration
        # shape: (batch, 1, dim_rain * 16) -> (batch, dim_rain * 16)
        lstm_out, _ = self.lstm2(lstm_out_1)
        rain_emb = lstm_out[:, -1, :].reshape((-1, self.dim_rain * 16))

        emb_cat = torch.cat([geo_emb, rain_emb], dim=1)
        logits = self.fc(emb_cat)
        return logits


