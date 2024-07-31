import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.nn.functional import normalize
from sklearn.preprocessing import MaxAbsScaler


GEO_FIELDS = [
    'H_mean', 'Slope_mean',
    'Aspect_mea', 'c_section_', 'c_surface_', 'cut_depth_', 'elev_var_m',
    'rdls_mean', 'rougth_mea', 'soa_mean', 'sos_mean', 'curvature_',
    'acc_mean', 'flowLength', 'downstream', 'upstream_L', 'code1_rati', 'code2_rati'
]

N_GEO_FEATURES = len(GEO_FIELDS)

def assure_folder_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalized_laplacian(adj, device=torch.device('cuda')):
    """ Implementation of normalized Laplacian calculation `L = D^(-1/2) A_tilde D^(-1/2)`
    """

    # sum debugging
    print('Calculating normalized Laplacian')

    # A_tilde: adjacency matrix with added self loop
    A_tilde = adj + torch.eye(adj.size(0))

    # D[i, i]: Summation of A_tilde along axis 1
    row_sum = A_tilde.sum(1)

    # D^(-1/2)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    del row_sum
    
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

    del d_inv_sqrt

    adj.to(device)
    d_mat_inv_sqrt.to(device)

    return adj.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt).cpu()


def load_adjacency_matrix(path_adj, n_slopeunits):
    df_adj = pd.read_csv(path_adj)

    adj = np.zeros((n_slopeunits, n_slopeunits), dtype=np.float32)

    for row in df_adj.itertuples():
        adj[row.src_allslopeid][row.nbr_allslopeid] = row.LENGTH

    scaler = MaxAbsScaler()
    adj = adj.reshape(-1, 1)
    adj = scaler.fit_transform(adj)
    adj = adj.reshape((n_slopeunits, n_slopeunits))
    return adj