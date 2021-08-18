import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, lr=0.001, device=torch.device('cpu')):
        self.device = device
        self.model = model.to(device)
        self.lr = lr

        self.loss = nn.HuberLoss(delta=3)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, folder, train, valid, epochs=30, batch_size=128):
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

        best_loss = float("inf")

        train_losses = []
        valid_losses = []
        for e in range(epochs):
            accu_loss = 0

            for (_, rain, geo, collapse) in train_loader:
                self.model.zero_grad()

                rain = rain.to(self.device)
                geo = geo.to(self.device)
                collapse = torch.sqrt(collapse).to(self.device) # deal with imbalance

                out = self.model(rain, geo)
                ls = self.loss(out, collapse)

                self.optim.zero_grad()
                ls.backward()
                self.optim.step()

                accu_loss += ls / len(train_loader)
            print(f'[Epoch: {e}] loss={accu_loss:.4f}')

            print('Validation:')
            valid_loss, _ = self.test(valid, batch_size=batch_size)

            # early stopping
            if valid_loss < best_loss:
                path_model = os.path.join(folder, 'model.pt')
                torch.save(self.model.state_dict(), path_model)
                best_loss = valid_loss

            train_losses.append(accu_loss)
            valid_losses.append(valid_loss)
        return train_losses, valid_losses


    def test(self, dataset, batch_size=128, printing=False, model_file=None):

        # if supplied, load the pretrained weights
        if model_file:
            self.model.load_state_dict(torch.load(model_file))


        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            accu_loss = 0
            pred = torch.Tensor().to(self.device)


            for (_, rain, geo, collapse) in loader:
                rain = rain.to(self.device)
                geo = geo.to(self.device)
                collapse = torch.sqrt(collapse).to(self.device) # deal with imbalance

                out = self.model(rain, geo)

                pred = torch.cat((pred, out), dim=0)

                if printing:
                    for i in range(len(out)):
                        print(f'out: {out[i].item()}, ans: {collapse[i].item()}')


                accu_loss += self.loss(out, collapse) / len(loader)

            print(f'Average MSE = {accu_loss:.4f}')

        return accu_loss, pred