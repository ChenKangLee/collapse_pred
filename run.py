import os
import torch
from torch.utils.data import random_split
from utils.dataset import DatasetFCU, DatasetPyramid
from utils.util import assure_folder_exist, N_GEO_FEATURES
from model.baseline import FCU
from model.TGCN import TGCN
from trainer.supervised import SupervisedTrainer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

N_SLOPEUNIT = 38915


def train_baseline():
    # use cuda whenever possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device:", device)


    # HYPERPARAM
    dataset_name = 'processedFCU_max_ws_6'
    experiment_name = 'weighted_labels'
    BATCH_SIZE = 35600
    N_EPOCH = 60        # numbers of epoch to train the model
    LR = 0.00003        # learning rate

    # define paths
    path_processed = os.path.join('data', dataset_name)
    path_fig       = os.path.join('figs', dataset_name)
    path_model     = os.path.join('trained_models', dataset_name)

    assure_folder_exist(path_fig)
    assure_folder_exist(path_model)

    model = FCU(dim_rain=2, dim_geo=N_GEO_FEATURES, device=device, dropout_rate=0.5)

    print('Loading baseline model dataset from', dataset_name)
    dataset = DatasetFCU(path_processed, years=range(102, 105), normalize=True, window_size=6)
    train, valid, test = random_split(dataset, [0.7, 0.15, 0.15])

    # get label distribution statistics, we are only calculating count for training set
    positive_count = dataset.collapse[train.indices].sum()
    # `pos_weight` is invertly correlated to the negative percentage
    pos_weight = torch.tensor([(len(train) - positive_count) / positive_count])
    print(f"    Percentage of positive training data {positive_count / len(train):4f}")

    loss  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    writer = SummaryWriter(comment=experiment_name)
    trainer = SupervisedTrainer(model, loss, writer, tag='', lr=LR, device=device)

    # train
    print('Begin Training...')
    train_loss, valid_loss = trainer.train(
        path_model,
        train,
        valid,
        epochs=N_EPOCH,
        batch_size=BATCH_SIZE,
        # inspect=f'predictions/{dataset_name}_{experiment_name}'
    )

    # check performance of best model
    # path_best = os.path.join(path_model, 'model_epoch_5.pt')
    # print('Testing...')
    # _, pred, ground_truth = trainer.test(test, batch_size=BATCH_SIZE, model_file=path_best)


def train_pyramid():
    # use cuda whenever possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device:", device)

    # HYPERPARAM
    dataset_name = 'processedPyramid_ws_6'
    experiment_name = 'weighted_labels'
    
    BATCH_SIZE = 8
    N_EPOCH = 120        # numbers of epoch to train the model
    LR = 0.00002        # learning rate

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

    print('Loading TGCN model dataset from', dataset_name)
    dataset = DatasetPyramid(path_processed, years=range(102, 105), normalize=True)
    train, valid, test = random_split(dataset, [0.7, 0.15, 0.15])

    # calculate pos_weight for the training set
    pos_weight = _pyramid_calc_ratio(train)
    

    loss  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    writer = SummaryWriter(comment=experiment_name)
    trainer = SupervisedTrainer(model, loss, writer, tag='', lr=LR, device=device)

    # train
    print('Begin Training...')
    train_loss, valid_loss = trainer.train(
        path_model,
        train,
        valid,
        epochs=N_EPOCH,
        batch_size=BATCH_SIZE
    )

    # check performance of best model
    # path_best = os.path.join(path_model, 'model_epoch_5.pt')
    # print('Testing...')
    # _, pred = trainer.test(test, batch_size=BATCH_SIZE, model_file=path_best)


def _pyramid_calc_ratio(subset: Subset):
    """ For the pyramid dataset since the data comes in structured form
        so we need extra steps to calculate the statistics
    """

    dataset_len = pos_count = 0
    for idx in subset.indices:
        dataset_len += N_SLOPEUNIT
        pos_count += subset.dataset.collapse[idx].sum()
    
    print(f"    Percentage of positive training data {pos_count / dataset_len:4f}")

    pos_weight = torch.tensor([(dataset_len - pos_count) / pos_count])
    return pos_weight


if __name__ == "__main__":
    train_pyramid()