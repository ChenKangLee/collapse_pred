import os
import torch
from utils.dataset import CollapseDataset
from utils.util import assure_folder_exist
from model.rnn import RNN
from trainer.training import Trainer
from plotting.plot import plot_loss, plot_pred


def train_rnn():

    # use cuda whenever possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device:", device)

    # hyperparam
    # TODO: replace with argparse
    slope_units = 6651
    training_interval = (94, 102)
    valid_interval = (102, 103)
    test_interval = (103, 105)
    batch_size = 128
    epoch = 80
    lr = 0.00003

    home = os.path.expanduser('~')
    path_processed = os.path.join(home, 'Documents', 'data', 'processed')
    path_fig       = os.path.join(home, 'Documents', 'figs')
    path_model     = os.path.join(home, 'Documents', 'models')
    
    assure_folder_exist(path_fig)
    assure_folder_exist(path_model)

    dataset_train = CollapseDataset(slope_units=slope_units, path=path_processed, interval=training_interval)
    dataset_valid = CollapseDataset(slope_units=slope_units, path=path_processed, interval=valid_interval)
    dataset_test  = CollapseDataset(slope_units=slope_units, path=path_processed, interval=test_interval)

    model = RNN(dim_rain=6, dim_geo=26, dim_hidden=128, device=device)
    trainer = Trainer(model, lr=lr, device=device)

    train_loss, valid_loss = trainer.train(path_model, dataset_train, dataset_valid, epochs=epoch, batch_size=batch_size)


    # check performance of best model
    path_best = os.path.join(path_model, 'model.pt')

    _, pred = trainer.test(dataset_train, batch_size=batch_size, model_file=path_best)
    plot_pred(torch.sqrt(dataset_train.collapse), pred.cpu(), filename=os.path.join(path_fig, 'MSE_train.png'))

    _, pred = trainer.test(dataset_valid, batch_size=batch_size, model_file=path_best)
    plot_pred(torch.sqrt(dataset_valid.collapse), pred.cpu(), filename=os.path.join(path_fig, 'MSE_valid.png'))
    
    print('Testing')
    _, pred = trainer.test(dataset_test, batch_size=batch_size, printing=True, model_file=path_best)
    plot_pred(torch.sqrt(dataset_test.collapse), pred.cpu(), filename=os.path.join(path_fig, 'MSE_test.png'))

    plot_loss(train_loss, valid_loss, filename=os.path.join(path_fig, 'losses.png'))

    return


if __name__ == "__main__":
    train_rnn()