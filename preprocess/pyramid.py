import os
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocess.base import PreprocessorBase, sliding_window_iter
from utils.util import GEO_FIELDS


class PreprocessorPyramid(PreprocessorBase):
    def __init__(self):
        super(PreprocessorPyramid, self).__init__()


    def load(self, path_root: str, interval=[102,106], window_size=6):

        self.n_entries = 0

        # raindata is indexed using `slopecode1`, we load lookup table to convert to `allslopeid`
        df_slopeid = pd.read_excel(os.path.join(path_root, 'slopeunit_id.xlsx'))

        # loop thru years in specified interval range
        for year in range(interval[0], interval[1] + 1):
            print(f"Processing geodata of year {year - 1}-{year}")

            ## ---------  geo data ----------
            path_geo_data = os.path.join(path_root, 'geodata', f'陳荖旗_geo_database_{year - 1}{year}year.xlsx')
            if not os.path.exists(path_geo_data):
                continue
            df_geo = pd.read_excel(path_geo_data)

            # just to be sure sort by allslopeid
            df_geo = df_geo.sort_values(['allslopeid'])

            # extract the geological infos and normalize
            geo_values = df_geo[GEO_FIELDS].to_numpy(dtype=np.float32)
            normalizer = StandardScaler().fit(geo_values)
            scaled_geo_value = normalizer.transform(geo_values)

            # raindata and collapse go hand-in-hand here
            # numpy array to aggregate entries accross rain events
            n_slopeunits = len(geo_values)
            aggregated_rain = np.empty((0, n_slopeunits, window_size, 2))
            aggregated_collapse = np.empty((0, n_slopeunits, 1))

            path_rain = os.path.join(path_root, 'eventRaindata', f'{year - 1}{year}')
            for y in year - 1, year:
                for eventID in string.ascii_uppercase:
                    # skip if event doesn't exist
                    path_event = os.path.join(path_rain, f'{y}{eventID}')
                    if not os.path.exists(path_event):
                        continue

                    print(f"    Processing rain event {y}{eventID}")

                    ## ---------  rain data ----------
                    df_I = pd.read_excel(os.path.join(path_event, 'I.xlsx'))
                    df_R = pd.read_excel(os.path.join(path_event, 'R.xlsx'))

                    # left join to get corresponding `allslopeid`
                    df_I = df_I.set_index('slopecode1').join(df_slopeid.set_index('slopecode1'), on='slopecode1', how='left')
                    df_R = df_R.set_index('slopecode1').join(df_slopeid.set_index('slopecode1'), on='slopecode1', how='left')

                    df_I = df_I.set_index('allslopeid').sort_index().reset_index().drop(columns='allslopeid')
                    df_R = df_R.set_index('allslopeid').sort_index().reset_index().drop(columns='allslopeid')

                    # TODO: try using max of convolution instead of last timestamp?
                    slideEventI = list(sliding_window_iter(df_I, window_size + 1))
                    slideEventR = list(sliding_window_iter(df_R, window_size + 1))

                    # each result of the sliding window creates a new entry (n_window = event_len - window_size)
                    # In the pyramid version we are preserving the spatial structure, so we stack along a new axis
                    # shape: (n_window, n_slopeunit, window_size + 1)
                    slideEventI = np.stack(slideEventI)
                    slideEventR = np.stack(slideEventR)

                    # remove the extra window width needed for collpase calculation
                    # shape: (n_window, n_slopeunit, window_size)
                    eventI = np.delete(slideEventI, -1, axis=2)
                    eventR = np.delete(slideEventR, -1, axis=2)

                    # shape: (n_window, n_slopeunit, window_size, 2)
                    raindata = np.stack((eventI, eventR), axis=-1)
                    # add to aggregation
                    aggregated_rain = np.concatenate((aggregated_rain, raindata), axis=0)

                    ## ---------  collpase data ----------
                    # We are projecting the collapse of end of each year onto each window
                    # here we added a hueristic where only if the raindata exceed a certain
                    # threshold do we register this as a collapse event
                    n_windows = len(df_I.columns) - window_size

                    # these originally has shape: (n_slopeunit, )
                    i_max = df_geo['imax'].to_numpy()
                    R_max = df_geo['R(imax)'].to_numpy()
                    did_collapse = (df_geo['add_3_4'] != 0).to_numpy()

                    # expand dimension on axis 0
                    # shape: (1, n_slopeunit)
                    i_max = np.expand_dims(i_max, axis=0)
                    R_max = np.expand_dims(R_max, axis=0)
                    did_collapse = np.expand_dims(did_collapse, axis=0)


                    # repeating on axis-0 to match the slideEvent shapes
                    # shape: (n_window, n_slopeunit, 1)
                    i_max = np.repeat(i_max, n_windows, axis=0)
                    R_max = np.repeat(R_max, n_windows, axis=0)
                    did_collapse = np.repeat(did_collapse, n_windows, axis=0)

                    i_rate = slideEventI[:, :, -1] / (i_max + self.epsilon)
                    R_rate = slideEventR[:, :, -1] / (R_max + self.epsilon)
                    
                    # we've selected the threshold to be `average of i_rate and R_rate > 0.7`
                    thresholded = (0.5 * i_rate + 0.5 * R_rate) > 0.7
                    calculated_collapse = (thresholded & did_collapse)
                    calculated_collapse = np.expand_dims(calculated_collapse, axis=-1)

                    # add to aggregation
                    aggregated_collapse = np.concatenate((aggregated_collapse, calculated_collapse), axis=0)

                    # stats keeping
                    self.n_entries += n_windows

            self.geo[year] = np.nan_to_num(scaled_geo_value).astype(np.float32)
            self.rain[year] = np.nan_to_num(aggregated_rain).astype(np.float32)
            self.collapse[year] = np.nan_to_num(aggregated_collapse).astype(np.float32)

    