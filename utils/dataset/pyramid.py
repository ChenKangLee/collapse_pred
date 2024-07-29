import os
import pickle
import numpy as np
from .base import DatasetBase


class DatasetPyramid(DatasetBase):
    def __init__(self, path, years, n_slopeunit=38915, window_size=6) -> None:
        super().__init__()

        self.rain = np.empty((0, n_slopeunit, window_size, 2), dtype=np.float32)
        self.geo = {}
        self.collapse = np.empty((0, n_slopeunit, 1), dtype=np.float32)
        self.n_slopeunit = n_slopeunit

        # for the geo data, we dont have to tile them at init, we can use a reversed
        # interval to year lookup to fetch the corresponding geo data and return tiled
        # data on the fly
        self.sample_intervals = []

        with open(os.path.join(path, 'collapse.pickle'), 'rb') as f:
            collapse = pickle.load(f)

        with open(os.path.join(path, 'rain.pickle'), 'rb') as f:
            rain = pickle.load(f)

        with open(os.path.join(path, 'geo.pickle'), 'rb') as f:
            geo = pickle.load(f)

        self._load(rain, geo, collapse, years)


    def _load(self, rain, geo, collapse, years):
        self.geo = geo

        for year in years:
            self.rain = np.concatenate((self.rain, rain[year]), axis=0)

            # log the interval for conversion
            if len(self.sample_intervals) == 0:
                self.sample_intervals.append(((0, self.rain[year].shape[0] + 1), year))
            else:
                prev_interval_end = self.sample_intervals[-1][1]
                self.sample_intervals.append(((prev_interval_end, prev_interval_end + self.rain[year].shape[0] + 1), year))

            self.collapse = np.concatenate((self.collapse, collapse[year]), axis=0)


    def _idx2year(self, index):
        for (start, end), year in self.sample_intervals:
            if start <= index < end:
                return year


    def __getitem__(self, index):
        year = self._idx2year(index)
        return index, self.rain[index], self.geo[year].astype(np.float32), self.collapse[index]