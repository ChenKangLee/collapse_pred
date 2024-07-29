import os
import pickle
import numpy as np
from .base import DatasetBase
from utils.util import N_GEO_FEATURES


class DatasetFCU(DatasetBase):
    def __init__(self, path, years=range(102, 107), n_slopeunit=38915, window_size=3, resample=None, normalize=False) -> None:
        super().__init__()

        self.rain = np.empty((0, window_size, 2), dtype=np.float32)
        self.geo = np.empty((0, N_GEO_FEATURES), dtype=np.float32)
        self.collapse = np.empty((0, 1), dtype=np.float32)
        self.n_slopeunit = n_slopeunit

        with open(os.path.join(path, 'collapse.pickle'), 'rb') as f:
            collapse = pickle.load(f)

        with open(os.path.join(path, 'rain.pickle'), 'rb') as f:
            rain = pickle.load(f)

        with open(os.path.join(path, 'geo.pickle'), 'rb') as f:
            geo = pickle.load(f)

        self._load(rain, geo, collapse, years, resample, normalize)


    def _load(self, rain, geo, collapse, years, resample, normalize):
        for year in years:
            self.rain = np.concatenate((self.rain, rain[year]), axis=0)
            self.geo = np.concatenate((self.geo, geo[year]), axis=0)
            self.collapse = np.concatenate((self.collapse, collapse[year]), axis=0)
        
        if resample:
            self.rain, self.geo, self.collapse = self._resample_dataset(self.rain, self.geo, self.collapse, resample)

        if normalize:
            # normalize features
            self.rain = self._normalize(self.rain)
            self.geo = self._normalize(self.geo)


    def __getitem__(self, index):
        return index, self.rain[index], self.geo[index], self.collapse[index]
