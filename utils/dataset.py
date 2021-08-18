import os
from numpy.core import overrides
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import math


class CollapseDataset(Dataset):
    def __init__(self, slope_units=6651, path=None, interval=None, neg_filter_rate=0):

        # the attributes of different data are stored in separate numpy arrays
        # all sharing the same indexing
        self.max_len = 0
        self.slope_units = slope_units # numbers of slope-units

        self.neg_filter_rate = neg_filter_rate

        if path:
            assert(interval)
            self.load(path, interval)


    def __len__(self):
        return self.collapse.shape[0]


    def __getitem__(self, idx):
        # slope_id, rain_seq, geodata, collapse
        return idx % self.slope_units, self.rain[idx], self.geodata[idx], self.collapse[idx]


    def load(self, path, interval=(94,107), neg_filter_rate=0):

        # pre-run thru raindata to determine max_len for padding
        for year in range(interval[0], interval[1]):
            path_rain = os.path.join(path, 'rain', f'year_{year}.npy')
            with open(path_rain, 'rb') as file:
                rain_arr = np.load(file)
                self.max_len = max(self.max_len, rain_arr.shape[1])


        # main loop for loading data
        rain = []
        geodata = []
        collapse = []
        label = []

        for year in range(interval[0], interval[1]):
            # raindata
            path_rain = os.path.join(path, 'rain', f'year_{year}.npy')
            with open(path_rain, 'rb') as file:
                rain_arr = np.load(file)

                # raindata is padded here to the global max in the interval
                rain_arr = np.pad(rain_arr, ((0,0), (0,self.max_len - rain_arr.shape[1]), (0, 0)))
                rain.append(rain_arr)

            # geodata
            path_geodata = os.path.join(path, 'geodata', f'year_{year}.npy')
            with open(path_geodata, 'rb') as file:
                geo_arr = np.load(file, allow_pickle=True)
                geodata.append(geo_arr)

            # collapse
            path_collapse = os.path.join(path, 'collapse', f'year_{year}.npy')
            with open(path_collapse, 'rb') as file:
                collapse_arr = np.load(file)
                collapse.append(collapse_arr)
                

        # pytorch has default type of float32
        self.rain = torch.from_numpy(np.concatenate(rain, axis=0)).float()
        self.geodata = torch.from_numpy(np.concatenate(geodata, axis=0)).float()
        self.collapse = torch.from_numpy(np.concatenate(collapse, axis=0)).float()

        print(f'Loaded {interval[1] - interval[0]} year(s) of data.')
        return


    def filter(self, neg_filter_rate):
        mask = torch.zeros(self.collapse.shape[0], dtype=torch.bool)
        for i in range(mask.shape[0]):
            if math.isclose(self.collapse[i], 0, abs_tol=0.000001) and random.random() < neg_filter_rate:
                mask[i] = 0
            else:
                mask[i] = 1

        self.rain = self.rain[mask]
        self.geodata = self.geodata[mask]
        self.collapse = self.collapse[mask]
        return