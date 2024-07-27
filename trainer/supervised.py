import os
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from utils.util import assure_folder_exist


class SupervisedTrainer:
    def __init__(self, model, loss, tag='', lr=0.001, device=torch.device('cpu')):
        self.device = device
        self.model = model.to(device)
        self.lr = lr
        self.tag = tag

        self.writer = SummaryWriter()

        self.loss = loss
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self, folder, train, valid, epochs=30, batch_size=128, inspect=None):
        """ Implements batched forwarding passing and backprop.

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

        for e in tqdm(range(epochs)):
            accu_loss = 0
        
            for (_, rain, geo, label) in tqdm(train_loader):
                self.model.zero_grad()

                rain = rain.to(self.device)
                geo = geo.to(self.device)
                label = label.to(self.device)

                out = self.model(rain, geo)
                ls = self.loss(out, label)

                self.optim.zero_grad()
                ls.backward()
                self.optim.step()

                accu_loss += ls.item()
            
            
            accu_loss /= len(train_loader)
            print(f'[Epoch: {e}] Training loss = {accu_loss:.4f}')
            self.writer.add_scalar(self.tag + '/loss/train', accu_loss, e)
            
            if e % 5 == 0:
                print(f'[Epoch: {e}] Validation:')
                valid_loss, logits = self.test(valid, batch_size=batch_size, log_metrics=True, epoch=e)
                self.writer.add_scalar(self.tag + '/loss/valid', valid_loss, e)

                if inspect:
                    assure_folder_exist(inspect)
                    np.savetxt(f'{inspect}/epoch_{e}_valid.txt', logits)

                # early stopping
                if valid_loss < best_loss:
                    path_model = os.path.join(folder, f'model_epoch_{e}.pt')
                    torch.save(self.model.state_dict(), path_model)
                    best_loss = valid_loss

                train_losses.append(accu_loss)
                valid_losses.append(valid_loss)
        return train_losses, valid_losses


    def test(self, dataset: Subset, batch_size=128, model_file=None, log_metrics=False, epoch=None):
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
            pred = torch.Tensor()

            for (_, rain, geo, label) in tqdm(loader):
                rain = rain.to(self.device)
                geo = geo.to(self.device)
                label = label.to(self.device)

                logits = self.model(rain, geo)
                probabilities = torch.sigmoid(logits).cpu()

                pred = torch.cat((pred, probabilities), dim=0)

                accu_loss += self.loss(logits, label).item()

            accu_loss /= len(loader)

            # extract ground truth labels
            label_true = dataset.collapse
            label_pred = (pred > 0.5).float()

            accuracy = accuracy_score(label_true, label_pred)
            precision = precision_score(label_true, label_pred)
            recall = recall_score(label_true, label_pred)
            macro_f1 = f1_score(label_true, label_pred, average='macro')

            print(f'loss = {accu_loss:.4f}')
            print(f'Accuracy =', accuracy)
            print(f'Precision =', precision)
            print(f'Recall =', recall)
            print(f'Macro F1 =', macro_f1)

            if log_metrics:
                self.writer.add_scalar(self.tag + '/metrics/accuracy', accuracy, epoch)
                self.writer.add_scalar(self.tag + '/metrics/precision', precision, epoch)
                self.writer.add_scalar(self.tag + '/metrics/recall', recall, epoch)
                self.writer.add_scalar(self.tag + '/metrics/macro_f1', macro_f1, epoch)


        return accu_loss, pred