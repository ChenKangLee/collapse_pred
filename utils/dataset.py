import os
import numpy as np
from sklearn.utils import resample
import torch
from collections import Counter
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from utils.util import assure_folder_exist


class CollapseDataset(Dataset):
    def __init__(self, slope_units=6651, path=None, interval=None, resample=None, label_bins=[0.0, 1.0]):

        # the attributes of different data are stored in separate numpy arrays
        # all sharing the same indexing
        self.max_len = 0
        self.slope_units = slope_units # numbers of slope-units

        self.resample = resample
        self.label_bins = label_bins

        if path:
            assert(interval)
            self._load(path, interval)


    def __len__(self):
        return self.label.shape[0]


    def __getitem__(self, idx):
        # slope_id, rain_seq, geodata, collapse
        return idx % self.slope_units, self.rain[idx], self.geodata[idx], self.label[idx]


    def _load(self, path, interval=(94,107)):
        """ Load the preprocessed data in the given interval. Train/test split
            can be done here.

            Parameters:
            -----------
            path (str):
                Directory that holds the preprocessed files

            interval ((int, int)):
                Interval of data to load. The range follows python convention, 
                (aka ending excluded)
        """

        # pre-run thru raindata to determine max_len for padding
        for year in range(interval[0], interval[1]):
            path_rain = os.path.join(path, 'rain', f'year_{year}.npy')
            with open(path_rain, 'rb') as file:
                rain_arr = np.load(file)
                self.max_len = max(self.max_len, rain_arr.shape[1])


        # main loop for loading data
        rain = []
        geodata = []
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
                label.append(np.digitize(collapse_arr, self.label_bins, right=True).reshape(-1)) # categorize by the supplied bins

        np_rain = np.concatenate(rain, axis=0)
        np_geodata = np.concatenate(geodata, axis=0)
        np_label = np.concatenate(label, axis=0)

        if self.resample:
            np_rain, np_geodata, np_label = self._resample(np_rain, np_geodata, np_label)

        # pytorch has default type of float32
        self.rain = torch.from_numpy(np_rain).float()
        self.geodata = torch.from_numpy(np_geodata).float()
        self.label = torch.from_numpy(np_label).type(torch.LongTensor)

        print(f'Loaded {interval[1] - interval[0]} year(s) of data.')
        return


    def _resample(self, rain, geodata, label):
        """ Resample the data according to the resampling method specified in `self.resample`
            (`rain` and `geodata` are features)
            
            Parameters:
            -----------
            rain (Numpy array):
                Rain data, shape = (dataset_size, max_len of event, 6)

            geodata (Numpy array):
                Geodata, shape = (dataset_size, 26)

            label (Numpy array):
                Collapse labels, shape = (dataset_size, )

            Returns:
            ---------
            The resampled result of the supplied data.
        """

        print(f"Original label distribution: {Counter(label)}")

        # join feature for resampling
        dataset_size = len(label)
        rain_dim = rain.shape[2]

        rain = rain.reshape(dataset_size, -1) # flatten for concatenating
        feature = np.concatenate([rain, geodata], axis=1)

        if self.resample == 'smote':
            print("Resampling using SMOTE...")
            sampler = SMOTE()
        elif self.resample == 'under':
            print("Resampling using RandomUnderSampler...")
            sampler = RandomUnderSampler(sampling_strategy='majority')
    
        feature_sampled, label_sampled = sampler.fit_resample(feature, label)
        print(f"Resampled label distribution: {Counter(label_sampled)}")
        
        splitted = np.split(feature_sampled, [rain.shape[1], feature_sampled.shape[1]], axis=1)
        rain = splitted[0].reshape(-1, self.max_len, rain_dim)
        geodata = splitted[1]
        label = label_sampled
        

        return rain, geodata, label