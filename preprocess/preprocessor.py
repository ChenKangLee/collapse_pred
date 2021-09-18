rimport os
import re
import string
from tokenize import Double
import numpy as np
import pandas as pd
from sklearn import preprocessing
from utils.util import assure_folder_exist

class Preprocessor:
    def __init__(self):
        self.rain = {}
        self.geodata = {}
        self.collapse = {}


    def load(self, path_root: str, interval, window_sizes):
        """ Loads all rainfall and collapse data in the range
            given by `interval`.

            Parameters:
            -----------
            path_root (str):
                The path to the raw input data (must follow naming scheme)
            
            interval (int, int):
                The interval (in years) we should preprocess. Note that this
                is the ENDING year, and the excludes the ending border.

            window_sizes (list if int):
                This specifies the size of the convolution kernel, this also effects the
                dimension of preprocessed raindata, with `dim = len(window_sizes)`

            Returns:
            --------
            None
        """
        for year in range(interval[0], interval[1]):
            self.rain[year] = self._load_rain(path_root, year, window_sizes=window_sizes)
            self.geodata[year] = self._load_geo(path_root, year)
            self.collapse[year] = self._load_collapse(path_root, year)

    
    def dump(self, path_dest: str):
        """ Save the processed data to disk.
            The data are separated into individual files by year.


            Parameter:
            ----------
            path_dest (str):
                The path to the output desination of the processed data.

            Return:
            -------
            None
        """

        path_rain = os.path.join(path_dest, 'rain')
        path_geodata = os.path.join(path_dest, 'geodata')
        path_collapse = os.path.join(path_dest, 'collapse')
        
        assure_folder_exist(path_rain)
        assure_folder_exist(path_geodata)
        assure_folder_exist(path_collapse)

        for year in self.collapse:
            path_rain_file = os.path.join(path_rain, f'year_{year}.npy')
            path_geo_file = os.path.join(path_geodata, f'year_{year}.npy')
            path_collapse_file = os.path.join(path_collapse, f'year_{year}.npy')

            with open(path_rain_file, 'wb') as file:
                np.save(file, self.rain[year])

            with open(path_geo_file, 'wb') as file:
                np.save(file, self.geodata[year])

            with open(path_collapse_file, 'wb') as file:
                np.save(file, self.collapse[year])
        return


    def _load_rain(self, path_root: str, year: int, window_sizes):
        """ Load the rainfall data in `year`

            Returns:
            --------
            Numpy array storing the rainfall data, with shape (# of units, # of event)
        """

        # Rain event data is represented with Imax * R at that time (TODO)
        path_rain = os.path.join(path_root, 'event', 'raindata')

        rains = []
        for subyear in [str(year - 1).zfill(3), str(year).zfill(3)]:
            for eventID in string.ascii_uppercase:
                path_i = os.path.join(path_rain, f'{year - 1}{year}year_{subyear}{eventID}_I.csv')
                path_r = os.path.join(path_rain, f'{year - 1}{year}year_{subyear}{eventID}_R.csv')

                # skip if file doesn't exist, the two file appears together, we only need to check one
                if not os.path.exists(path_i):
                    continue

                print('Loaded rain data from:', path_i)

                np_i = np.genfromtxt(path_i, delimiter=',', skip_header=1)
                np_i = np.nan_to_num(np_i) # seems like there are missing values, which results in NaN
                np_i = np.transpose(np_i)[1:, :] # skipping the first column

                np_r = np.genfromtxt(path_r, delimiter=',', skip_header=1)
                np_r = np.nan_to_num(np_r)
                np_r = np.transpose(np_r)[1:, :]

                rains.append(self.collate_event_rain(np_i, np_r, window_sizes=window_sizes))
        
        res = np.stack(rains, axis=1) # debug
        return res


    def collate_event_rain(self, np_i, np_r, window_sizes=[1]):
        """ Here, we generate X by convolving the raw rain data.

            Parameters:
            -----------
            np_i (Numpy array):
                The intensity of the rainfall in a given event.

            np_r (Numpy array):
                The r value of the event.

            Return:
            --------
            Extracted value that represents the event.
        """
        
        collated = []

        for window_size in window_sizes:
            np_i_convolved = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size), mode='valid'), axis=1, arr=np_i)
            np_r_convolved = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size), mode='valid'), axis=1, arr=np_r)
        
            max_index = np.argmax(np_i_convolved, axis=1) # find the index with the maximum intensity
            rows  = np.arange(len(max_index))   # for indexing the 2D R array

            collated.append(np_i_convolved[rows, max_index])
            collated.append(np_r_convolved[rows, max_index])
        return np.stack(collated, axis=1)


    def _load_geo(self, path_root: str, year: int):
        """ Load the geodata in `year`.
            The geodata also includes the `recovered collapse` (code 1) and `pre-existing collapse` (code 2) of the collision data

            Returns:
            --------
            Numpy array with shape (# of slope-units, 20)
        """


        ##### geodata (total of 17 entries) #####
        path_geodata = os.path.join(path_root, 'geodata', 'geodata.csv')
        df_geodata = pd.read_csv(path_geodata).transpose()
        df_geodata.columns = df_geodata.iloc[0]
        df_geodata.drop(df_geodata.index[0], inplace=True)

        # single out the categorical data, since we won't be normalizing them
        # instead we will convert them to one-hot encoding
        df_gtype = pd.get_dummies(df_geodata['G_type'], prefix='G_type')


        # normalize geodata across slope-units before returning, currently using MaxAbsScalling
        df_geodata.drop(columns=['G_type'], inplace=True)
        geodata_values = df_geodata.to_numpy()
        transformer = preprocessing.MaxAbsScaler().fit(geodata_values)
        scaled_value = transformer.transform(geodata_values)


        ##### ndvi data (changes year-to-year) #####
        path_ndvi = os.path.join(path_root, 'geodata', 'ndvi.csv')
        df_ndvi = pd.read_csv(path_ndvi).transpose()
        df_ndvi.columns = df_ndvi.iloc[0]
        df_ndvi.drop(df_ndvi.index[0], inplace=True)


        ##### code2 collapse data #####
        # This also changes on a yearly basis, append to geodata
        path_collapse = os.path.join(path_root, 'collapse', f'CollapseData_{year - 1}{year}.csv')
        df_collapse = pd.read_csv(path_collapse).transpose()
        df_collapse.columns = df_collapse.iloc[0]
        df_collapse.drop(df_collapse.index[0], inplace=True)
        df_collapse_code1 = df_collapse['code1_rati'] * 0.01 # the unit is percentage
        df_collapse_code2 = df_collapse['code2_rati'] * 0.01 

        return np.concatenate([
            scaled_value,
            df_gtype.to_numpy(),
            df_ndvi[f'{year - 1}ndvi_m'].to_numpy()[:, np.newaxis],
            df_collapse_code1.to_numpy()[:, np.newaxis],
            df_collapse_code2.to_numpy()[:, np.newaxis]
        ], axis=1).astype(float)



    def _load_collapse(self, path_root: str, year: int):
        """ Load the collapse data (the label) in `year`

            Returns:
            --------
            Numpy array storing the rainfall data, with shape (# of slope-units,)
        """

        path_collapse = os.path.join(path_root, 'collapse', f'CollapseData_{year - 1}{year}.csv')
        df_collapse = pd.read_csv(path_collapse)

        df_collapse = df_collapse[df_collapse['slope_id'] == 'add_3_4'] # extract the row `add_3_4`
        df_collapse.drop(columns=['slope_id'], inplace=True)
        df_collapse.fillna(0, inplace=True)

        return df_collapse.transpose().to_numpy() * 0.01 # the value is in percentage


def run():
    home = os.path.expanduser('~')
    root = os.path.join(home, 'Documents', 'data', 'F')
    dest = os.path.join(home, 'Documents', 'data', 'processed')

    interval = (94, 107)

    preprocessor = Preprocessor()
    preprocessor.load(root, interval, window_sizes=[1,3,6]
    )

    preprocessor.dump(dest)
    return


if __name__ == "__main__":
    run()