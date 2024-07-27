import os
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocess.base import PreprocessorBase, sliding_window_iter
from utils.util import GEO_FIELDS


class PreprocessorFCU(PreprocessorBase):
    def __init__(self):
        super(PreprocessorFCU, self).__init__()


    def load(self, path_root: str, interval=[102,106], window_size=3):
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
            aggregated_rain = np.empty((0, window_size, 2))
            aggregated_collapse = np.empty((0, 1))

            path_rain = os.path.join(path_root, 'eventRaindata', f'{year - 1}{year}')

            # this tracks the accumulated count of windows that we got out of this
            # this determines how many times we will tile the geo data in the end
            total_repeat = 0
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

                    slideEventI = list(sliding_window_iter(df_I, window_size + 1))
                    slideEventR = list(sliding_window_iter(df_R, window_size + 1))

                    # flatten spatially since in the FCU model they are not considering
                    # the spatial relationship
                    # shape: (number of total collate * num of slopeunit, window_size + 1)
                    slideEventI = np.vstack(slideEventI)
                    slideEventR = np.vstack(slideEventR)

                    # remove the extra window width needed for collpase calculation
                    # shape: (number of total collate * num of slopeunit, window_size)
                    eventI = np.delete(slideEventI, -1, axis=1)
                    eventR = np.delete(slideEventR, -1, axis=1)

                    # shape: (number of total collate * num of slopeunit, window_size, 2)
                    raindata = np.dstack((eventI, eventR))
                    # add to aggregation
                    aggregated_rain = np.vstack((aggregated_rain, raindata))

                    ## ---------  collpase data ----------
                    # We are projecting the collapse of end of each year onto each window
                    # here we added a hueristic where only if the raindata exceed a certain
                    # threshold do we register this as a collapse event

                    # we need to match the shape of the geo data to the flattened sliding window result
                    n_windows = len(df_I.columns) - window_size
                    df_geo_repeated = pd.concat([df_geo] * n_windows, ignore_index=True)

                    i_max = df_geo_repeated['imax']
                    R_max = df_geo_repeated['R(imax)']
                    did_collapse = df_geo_repeated['add_3_4'] != 0

                    i_rate = slideEventI[:, -1] / (i_max + self.epsilon)
                    R_rate = slideEventR[:, -1] / (R_max + self.epsilon)
                    
                    # we've selected the threshold to be `average of i_rate and R_rate > 0.7`
                    thresholded = (0.5 * i_rate + 0.5 * R_rate) > 0.7
                    df_calculated_collapse = (thresholded & did_collapse)
                    calculated_collapse = df_calculated_collapse.to_numpy().reshape(-1, 1)

                    # add to aggregation
                    aggregated_collapse = np.vstack((aggregated_collapse, calculated_collapse))

                    # stat keeping
                    self.n_entries += aggregated_rain.shape[0]
                    total_repeat += n_windows

            # tile geo data accordingly
            geo_tiled = np.tile(scaled_geo_value, (total_repeat, 1))

            self.geo[year] = np.nan_to_num(geo_tiled).astype(np.float32)
            self.rain[year] = np.nan_to_num(aggregated_rain).astype(np.float32)
            self.collapse[year] = np.nan_to_num(aggregated_collapse).astype(np.float32)