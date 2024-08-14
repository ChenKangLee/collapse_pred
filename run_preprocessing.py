import os
import pandas as pd
from preprocess import PreprocessorPyramid, PreprocessorFCU


if __name__ == '__main__':
    # subsample slopeunits
    path_subset = os.path.join('data', 'subsampling', 'subsample_area.csv')
    subset = pd.read_csv(path_subset)

    preprocessor = PreprocessorFCU()
    preprocessor.load('data', interval=[102,106], window_size=6, df_subset=subset)
    preprocessor.dump('data/processedFCU_subsample_ws_6')

    print(f"Complete preprocessing for FCU data with {preprocessor.n_entries} entries.")
    del preprocessor


    preprocessor = PreprocessorPyramid()
    preprocessor.load('data', interval=[102,106], window_size=6, df_subset=subset)
    preprocessor.dump('data/processedPyramid_subsample_ws_6')

    print(f"Complete preprocessing for Pyramid data with {preprocessor.n_entries} entries.")
    