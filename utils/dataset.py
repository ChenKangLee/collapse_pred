import numpy as np
from torch.utils.data import Dataset
from utils.util import N_GEO_FEATURES


class FCUDataset(Dataset):
    def __init__(self, rain, geo, collapse, n_slopeunit=38915, window_size=3) -> None:
        super().__init__()

        self.rain = np.empty((0, window_size, 2))
        self.geo = {}
        self.collapse = np.empty((0, n_slopeunit, 1))
        self.n_slopeunit = n_slopeunit

        self._load(rain, geo, collapse)


    def _load(self, rain, geo, collapse):
        for year in rain:
            self.rain = np.concatenate((self.rain, rain[year]), axis=0)
            self.geo = np.concatenate((self.geo, geo[year]), axis=0)
            self.collapse = np.concatenate((self.collapse, collapse[year]), axis=0)


    def __len__(self):
        return self.rain.shape[0]
    

    def __getitem__(self, index):
        return index % self.n_slopeunit, self.rain[index], self.geo[index], self.collapse[index]



class PyramidDataset(Dataset):
    def __init__(self, rain, geo, collapse, n_slopeunit=38915, window_size=6) -> None:
        super().__init__()

        self.rain = np.empty((0, n_slopeunit, window_size, 2))
        self.geo = np.empty((0, n_slopeunit, N_GEO_FEATURES))
        self.collapse = np.empty((0, n_slopeunit, 1))
        self.n_slopeunit = n_slopeunit

        # for the geo data, we dont have to tile them at init, we can use a reversed
        # interval to year lookup to fetch the corresponding geo data and return tiled
        # data on the fly
        self.sample_intervals = []

        self._load(rain, geo, collapse)


    def _load(self, rain, geo, collapse):
        self.geo = geo

        for year in rain:
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


    def __len__(self):
        return self.rain.shape[0]
    

    def __getitem__(self, index):
        year = self._idx2year(index)
        return index, self.rain[index], self.geo[year], self.collapse[index]

    