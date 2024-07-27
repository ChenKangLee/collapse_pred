import os
import pickle
import abc
from utils.util import assure_folder_exist

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

        # here we are defining a small value that we add to the division
        # to prevent division by zero errors that produces NaN values
        # Impact of Epsilon & reasoning:
        # When historical max is zero: This adjustment makes the denominator very small but non-zero, which can result in a very large PR value. This is typically a desirable behavior because any non-zero event value compared to a historical max of zero indicates a significant event.
        # When historical max is non-zero: Adding epsilon has a negligible effect because the historical max is much larger than epsilon.
        self.epsilon = 1e-10

    @abc.abstractmethod
    def load(self, path_root: str, interval=[102,106], window_size=3):
        pass

    def dump(self, path_processed):
        assure_folder_exist(path_processed)
        for tag, data in zip(['rain', 'geo', 'collapse'], [self.rain, self.geo, self.collapse]):
            path = os.path.join(path_processed, f'{tag}.pickle')
            with open(path, 'wb') as f:
                pickle.dump(data, f)
