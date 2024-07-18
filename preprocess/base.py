import os
import pickle
import abc
from utils.util import assure_folder_exist

GEO_FIELDS = [
    'H_mean', 'Slope_mean',
    'Aspect_mea', 'c_section_', 'c_surface_', 'cut_depth_', 'elev_var_m',
    'rdls_mean', 'rougth_mea', 'soa_mean', 'sos_mean', 'curvature_',
    'acc_mean', 'flowLength', 'downstream', 'upstream_L', 'code1_rati', 'code2_rati'
]

def sliding_window_iter(series, size):
    """series is a column of a dataframe"""
    for start_col in range(len(series.columns) - size + 1):
        yield series.iloc[:, start_col:start_col + size]


class PreprocessorBase:
    def __init__(self):
        self.n_entries = 0
        self.rain = {}
        self.geo = {}
        self.collapse = {}

    @abc.abstractmethod
    def load(self, path_root: str, interval=[102,106], window_size=3):
        pass

    def dump(self, path_processed):
        assure_folder_exist(path_processed)
        for tag, data in zip(['rain', 'geo', 'collapse'], [self.rain, self.geo, self.collapse]):
            path = os.path.join(path_processed, f'{tag}.pickle')
            with open(path, 'wb') as f:
                pickle.dump(data, f)
