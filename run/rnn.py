import os
import torch
from utils.dataset import CollapseDataset
from utils.util import assure_folder_exist
from model.rnn import RNN
from trainer.supervised import SupervisedTrainer
from visualize.continuous import plot_loss
from visualize.categorical import plot_confusion


def train_rnn():

    # use cuda whenever possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device:", device)

    # hyperparam
    # TODO: replace with argparse?
    slope_units = 6651             # number of slope units in the dataset
    label_bins = [0.0, 0.02, 0.05] # this defines the bound of categories.
    training_interval = (94, 102)  # the years of data we should use as training
    valid_interval = (102, 103)    # the years of data we should use as validation
    test_interval = (103, 105)     # the years of data we should use as testing
    batch_size = 128 
    epoch = 120     # numbers of epoch to traing the model
    lr = 0.00003    # learning rate


    # define paths
    home = os.path.expanduser('~')
    path_processed = os.path.join(home, 'Documents', 'data', 'processed')
    path_fig       = os.path.join(home, 'Documents', 'figs')
    path_model     = os.path.join(home, 'Documents', 'models')
    
    assure_folder_exist(path_fig)
    assure_folder_exist(path_model)

    # load the dataset
    dataset_train = CollapseDataset(slope_units=slope_units, path=path_processed, interval=training_interval, label_bins=label_bins, resample='smote')
    dataset_valid = CollapseDataset(slope_units=slope_units, path=path_processed, interval=valid_interval, label_bins=label_bins)
    dataset_test  = CollapseDataset(slope_units=slope_units, path=path_processed, interval=test_interval, label_bins=label_bins)

    # initiate models and helpers
    model = RNN(dim_rain=6, dim_geo=26, n_labels=len(label_bins) + 1, dim_hidden=128, device=device)
    loss  = torch.nn.CrossEntropyLoss()
    trainer = SupervisedTrainer(model, loss, lr=lr, device=device)

    # train
    train_loss, valid_loss = trainer.train(path_model, dataset_train, dataset_valid, epochs=epoch, batch_size=batch_size)

    # check performance of best model
    path_best = os.path.join(path_model, 'model.pt')


    # plot the output of the best performing model
    _, pred = trainer.test(dataset_train, batch_size=batch_size, model_file=path_best)
    plot_confusion(len(label_bins) + 1, dataset_train.label, pred.cpu(), filename='training')

    _, pred = trainer.test(dataset_valid, batch_size=batch_size, model_file=path_best)
    plot_confusion(len(label_bins) + 1, dataset_valid.label, pred.cpu(), filename='valid')
    
    print('Testing')
    _, pred = trainer.test(dataset_test, batch_size=batch_size, model_file=path_best)
    plot_confusion(len(label_bins) + 1, dataset_test.label, pred.cpu(), filename='testing')

    plot_loss(train_loss, valid_loss, filename=os.path.join(path_fig, 'losses.png'))

    return


if __name__ == "__main__":
    train_rnn()