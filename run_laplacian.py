import torch
from sklearn.preprocessing import MaxAbsScaler
from utils.util import normalized_laplacian, load_adjacency_matrix

N_SLOPEUNIT = 1464

path_subset = 'data/subsampling/subsample_area.csv'
path_adj = 'data/subsampling/subsample_adj.csv'

adj = load_adjacency_matrix(path_subset, path_adj, N_SLOPEUNIT)
adj = torch.from_numpy(adj)

laplacian = normalized_laplacian(adj)

torch.save(laplacian, 'data/laplacian_subsampled.pt')