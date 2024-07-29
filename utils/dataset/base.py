import numpy as np
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler


class DatasetBase(Dataset):
    def __init__(self):
        pass

    def _resample_dataset(self, rain, geo, collapse, resample):
        """ Resample the data according to the resampling method specified in `self.resample`
            (`rain` and `geodata` are features)
            
            Parameters:
            -----------
            rain (Numpy array):
                Rain data, shape = (dataset_size, max_len of event, 6)

            geodata (Numpy array):
                Geodata, shape = (dataset_size, 26)

            collapse (Numpy array):
                Collapse labels, shape = (dataset_size, )

            Returns:
            ---------
            The resampled result of the supplied data.
        """

        if resample == 'over':
            print("Oversampling using SMOTE...")
            sampler = SMOTE()
        elif resample == 'under':
            print("Undersampling using RandomUnderSampler")
            sampler = RandomUnderSampler(sampling_strategy='majority')

        print(f"Total number of samples:", len(collapse))
        print(f"Original percentage of collapses: {collapse.sum() / len(collapse)}")

        # record original shape of features
        dataset_size = rain.shape[0]
        shape_rain = rain.shape
        shape_geo = geo.shape

        rain = rain.reshape(dataset_size, -1) # flatten for concatenating
        geo = geo.reshape(dataset_size, -1)
        feature = np.concatenate([rain, geo], axis=1)

        feature_sampled, label_sampled = sampler.fit_resample(feature, collapse)
        print(f"Total number of samples:", len(label_sampled))
        print(f"Resampled label distribution: {label_sampled.sum() / len(label_sampled)}")
        
        # reconstruct the flattened sampled features back to the original shape of `rain` and `geo`
        rain_flattened_len = np.prod(shape_rain[1:])
        splitted = np.split(feature_sampled, [rain_flattened_len], axis=1)
        rain = splitted[0].reshape((-1,) + shape_rain[1:])
        geo = splitted[1].reshape((-1,) + shape_geo[1:])
        label = np.expand_dims(label_sampled, axis=1)

        return rain, geo, label
    

    def _normalize(self, inputs):
        """ Normalize `inputs` by sklearn.StandardScaler. We are saving the normalization
            until this step to preserve the original data as much as possible
        """

        scaler = StandardScaler()
        
        # record original shape of `inputs`
        original_shape = inputs.shape

        # flatten into (n_sample, n_features)
        inputs = inputs.reshape(original_shape[0], -1)

        normalized = scaler.fit_transform(inputs)

        # restore original shape
        normalized = normalized.reshape(original_shape)
        return normalized


    def __len__(self):
        return self.rain.shape[0]