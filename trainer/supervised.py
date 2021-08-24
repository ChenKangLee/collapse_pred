import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


class SupervisedTrainer:
    def __init__(self, model, loss, lr=0.001, device=torch.device('cpu')):
        self.device = device
        self.model = model.to(device)
        self.lr = lr

        self.loss = loss
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self, folder, train, valid, epochs=30, batch_size=128):
        """ Implements supervised training.

            Parameters:
            ------------
            folder (str):
                Folder to which the model weight file is stored.

            train/valid (CollapseDataset):
                Trainig/validation dataset.

            epoch (int):
                Numbers of epochs to train the model

            batch_size (int):
                Batch size

            Returns:
            --------
            train_losses (list of float):
                The losses from each epoch of training.

            valid_losses (list of float):
                The loss of the model from each epoch on the validation set.
        """

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

        best_loss = float("inf")
        train_losses = []
        valid_losses = []
        for e in range(epochs):
            accu_loss = 0

            for (_, rain, geo, label) in train_loader:
                self.model.zero_grad()

                rain = rain.to(self.device)
                geo = geo.to(self.device)
                label = label.to(self.device)

                out = self.model(rain, geo)
                ls = self.loss(out, label)

                self.optim.zero_grad()
                ls.backward()
                self.optim.step()

                accu_loss += ls / len(train_loader)
            print(f'[Epoch: {e}] Training loss={accu_loss:.4f}')

            print(f'[Epoch: {e}] Validation:')
            valid_loss, _ = self.test(valid, batch_size=batch_size)

            # early stopping
            if valid_loss < best_loss:
                path_model = os.path.join(folder, 'model.pt')
                torch.save(self.model.state_dict(), path_model)
                best_loss = valid_loss

            train_losses.append(accu_loss)
            valid_losses.append(valid_loss)
        return train_losses, valid_losses


    def test(self, dataset, batch_size=128, model_file=None):
        """ Generates prediction for a given dataset. Also records the loss.
            The operations' gradients are not recorded.

            Parameters:
            -----------
            dataset (CollapseDataset):
                The dataset we wish to test `self.model` on.

            batch_size (int):
                Batch size

            model_file (str):
                Path to the pretrained model file. If supplied, will load the
                weight to overwrite the model in `self.model`

            
            Returns:
            --------
            accu_loss (float):
                Accumulated loss of the model across the dataset.

            pred (torch.Tensor):
                The models prediciton for each example in the dataset.
        """


        # if supplied, load the pretrained weights
        if model_file:
            self.model.load_state_dict(torch.load(model_file))

        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            accu_loss = 0
            pred = torch.Tensor().to(self.device)

            for (_, rain, geo, label) in loader:
                rain = rain.to(self.device)
                geo = geo.to(self.device)
                label = label.to(self.device)

                out = self.model(rain, geo)
                pred = torch.cat((pred, out), dim=0)

                accu_loss += self.loss(out, label) / len(loader)

            label_true = dataset.label
            label_pred = torch.argmax(pred, dim=1).cpu()
            print(f'loss = {accu_loss:.4f}')
            print(f'microF1 =', f1_score(label_true, label_pred, average='micro'))

        return accu_loss, pred