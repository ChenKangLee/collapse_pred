import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import random_split
from utils.dataset import DatasetFCU, DatasetPyramid
from utils.util import assure_folder_exist, load_adjacency_matrix, N_GEO_FEATURES
from model.baseline import FCU
from model.TGCN import TGCN
from trainer.supervised import SupervisedTrainer


def train_baseline():
    # use cuda whenever possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device:", device)

    # HYPERPARAM
    dataset_name = 'processedFCU'
    N_SLOPEUNIT = 38915
    BATCH_SIZE = 12800
    N_EPOCH = 300       # numbers of epoch to train the model
    LR = 0.000001        # learning rate

    # define paths
    path_processed = os.path.join('data', dataset_name)
    path_fig       = os.path.join('figs', dataset_name)
    path_model     = os.path.join('trained_models', dataset_name)

    assure_folder_exist(path_fig)
    assure_folder_exist(path_model)

    print('Loading baseline model dataset from', dataset_name)
    # for ease of operation, we are using year 102-104 as training, 105 as validation and 106 as test
    train = DatasetFCU(path_processed, years=range(102, 105), resample='under', normalize=True)
    valid = DatasetFCU(path_processed, years=range(105, 106), resample='under', normalize=True)
    # test = DatasetFCU(path_processed, years=range(106, 107), normalize=True)


    model = FCU(dim_rain=2, dim_geo=N_GEO_FEATURES, device=device, dropout_rate=0.5)
    loss  = torch.nn.BCEWithLogitsLoss()
    trainer = SupervisedTrainer(model, loss, tag='FCU', lr=LR, device=device)

    # train
    print('Begin Training...')
    train_loss, valid_loss = trainer.train(path_model, train, valid, epochs=N_EPOCH, batch_size=BATCH_SIZE, inspect=f'predictions/{dataset_name}')

    # check performance of best model
    # path_best = os.path.join(path_model, 'model_epoch_5.pt')
    # print('Testing...')
    # _, pred = trainer.test(test, batch_size=BATCH_SIZE, model_file=path_best)


def train_pyramid():
    # use cuda whenever possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device:", device)

    # HYPERPARAM
    dataset_name = 'processedPyramid_ws_6'
    N_SLOPEUNIT = 38915
    BATCH_SIZE = 32
    N_EPOCH = 100       # numbers of epoch to train the model
    LR = 0.00001        # learning rate

    # define paths
    path_processed = os.path.join('data', dataset_name)
    path_fig       = os.path.join('figs', dataset_name)
    path_model     = os.path.join('trained_models', dataset_name)
    path_laplacian = os.path.join('data', 'laplacian.pt')

    assure_folder_exist(path_fig)
    assure_folder_exist(path_model)

    # load pre-calculated laplacian
    laplacian = torch.load(path_laplacian)

    model = TGCN(dim_rain=2, dim_geo=N_GEO_FEATURES, n_slopeunits=N_SLOPEUNIT, laplacian=laplacian, device=device, dropout_rate=0.5)
    loss  = torch.nn.BCEWithLogitsLoss()
    trainer = SupervisedTrainer(model, loss, tag='Pyramid', lr=LR, device=device)

    print('Loading baseline model dataset from', dataset_name)
    # for ease of operation, we are using year 102-104 as training, 105 as validation and 106 as test
    train = DatasetPyramid(path_processed, years=range(102, 105), normalize=True)
    valid = DatasetPyramid(path_processed, years=range(105, 106), normalize=True)

    # train
    print('Begin Training...')
    train_loss, valid_loss = trainer.train(path_model, train, valid, epochs=N_EPOCH, batch_size=BATCH_SIZE, inspect=f'predictions/{dataset_name}')

    # check performance of best model
    # path_best = os.path.join(path_model, 'model_epoch_5.pt')
    # print('Testing...')
    # _, pred = trainer.test(test, batch_size=BATCH_SIZE, model_file=path_best)


if __name__ == "__main__":
    train_baseline()