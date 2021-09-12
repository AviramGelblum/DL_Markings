import pandas as pd
import numpy as np
import math
import partitioning
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from enum import Enum, auto
from Pipeline import Iprocessable
import itertools


class DataType(Enum):
    Full = auto()
    Training = auto()
    Validation = auto()
    Test = auto()


class Data(Iprocessable):

    def __init__(self, data: list, runner):
        self.labels = [mat[:, -1] for mat in data]  # assume numpy
        self.features = [mat[:, :-1] for mat in data]

        self.windowed_labels = None
        self.windowed_features = None
        self.association = None

        self._type = DataType.Full

        self.stride = runner.model_parameters['stride']
        self.dilation_window_size = runner.model_parameters['dilation_window_size']
        self.sample_window_size = runner.model_parameters['sample_window_size']
        self.percent_test = runner.training_parameters['percent_test']
        self.number_of_folds = runner.training_parameters['number_of_folds']

        self.test_trajectory_indices = None
        self.test_windowed_indices = None
        self.window_indices_in_folds = []
        self.augmentation_random_thetas = []
        self.history = runner.history

    def from_existing(self, indices, type_in: DataType):
        new_data = copy.copy(self)
        new_data.windowed_labels = new_data.windowed_labels[indices]
        new_data.windowed_features = new_data.windowed_features[indices]
        new_data.association = new_data.association[indices]

        associated_trajectory_indices = np.unique(new_data.association)
        new_data.features = [self.features[k] for k in associated_trajectory_indices]
        new_data.labels = [self.labels[k] for k in associated_trajectory_indices]

        if type_in is None:
            type_in = DataType.Full
        if not isinstance(type_in, DataType):
            raise TypeError('type_in must be a DataType Enum')

        new_data._type = type_in
        return new_data

    def verify_type(self, instance_types):
        try:
            _ = iter(instance_types)
        except TypeError:
            return True

        if self._type in instance_types:
            return True
        return False

    def remove_features(self, to_remove):
        num_features = self.features[0].shape[1]
        to_keep = list(set(range(num_features))-set(to_remove))
        self.features = [mat[:, to_keep] for mat in self.features]
        self.history['features_removed'] = to_remove

    def transform_features(self, indices_to_transform, transformer=StandardScaler):
        for i in indices_to_transform:
            stscaler = transformer()
            only_feature_list = [p[:, i] for p in self.features]
            only_feature_array = np.concatenate(only_feature_list)
            stscaler.fit(only_feature_array.reshape(-1, 1))
            for sequence in self.features:
                sequence[:, i] = np.squeeze(stscaler.transform(sequence[:, i].reshape(-1, 1)))
        self.history['features_transformed'] = indices_to_transform
        self.history['transformer'] = transformer.__name__

    def get_only_positive_examples(self):
        num_positives = np.array([sum(l) for l in self.labels])
        where_positives = np.nonzero(num_positives)[0]
        self.labels = [self.labels[w] for w in where_positives]
        self.features = [self.features[w] for w in where_positives]
        self.history['removed no-positive sequences'] = True

    def assure_binary_labels(self):
        for sequence in self.labels:
            sequence[np.greater(sequence, 1)] = 1
        self.history['assured binary labels'] = True

    def cut_sequence_into_windows(self):
        # Cut data and labels into overlapping windows of size window_size, with stride stride.
        # Zero-pad samples with size<window_size

        label_list = []
        feature_list = []
        traj_index_list = []
        count = 0
        for feature_sequence, label_sequence in zip(self.features, self.labels):
            length = feature_sequence.shape[0]
            last_full_window_ind = max(math.floor((length - self.sample_window_size) / self.stride + 1), 1)
            seq_feature_list = []
            seq_label_list = []
            for i in range(0, length, self.stride):
                seq_feature_list.append(feature_sequence[i: i + self.sample_window_size, :])
                seq_label_list.append(label_sequence[i: i + self.sample_window_size])

            # remove last part to make sure sizes are the same
            seq_feature_list = seq_feature_list[:last_full_window_ind]
            seq_label_list = seq_label_list[:last_full_window_ind]

            if last_full_window_ind == 1:  # Zero-pad samples with size<window_size
                seq_feature_list[0] = np.pad(seq_feature_list[0],
                                             ((0, self.sample_window_size - seq_feature_list[0].shape[0]), (0, 0)))
                seq_label_list[0] = np.pad(seq_label_list[0], (0, self.sample_window_size - seq_label_list[0].shape[0]))

            feature_list.extend(seq_feature_list)
            label_list.extend(seq_label_list)
            traj_index_list.extend([count] * len(seq_feature_list))
            count += 1

        self.windowed_labels = np.stack(label_list).astype(float)
        self.windowed_features = np.stack(feature_list).astype(float)
        self.association = np.array(traj_index_list)
        self.history['windowed data'] = True

    def split_test_data(self):
        original_trajectory_sizes = [f.shape[0] for f in self.features]
        total_test_trajectory_length = round(self.percent_test*sum(original_trajectory_sizes)/100)
        max_length = round(self.percent_test*len(original_trajectory_sizes)/100)
        _, self.test_trajectory_indices = partitioning.find_approximate_subset(
            original_trajectory_sizes, list(range(len(original_trajectory_sizes))), total_test_trajectory_length, max_length)
        self.test_windowed_indices = np.where(np.isin(self.association, self.test_trajectory_indices))[0]
        self.history['split test data'] = True

    def stratify_training_data(self):
        post_test_trajectory_sizes = [f[1].shape[0] for f in enumerate(self.features) if f[0] not in
                                      self.test_trajectory_indices]
        post_test_number_of_positives = [sum(l[1]) for l in enumerate(self.labels) if l[0] not in
                                         self.test_trajectory_indices]

        scaler = MinMaxScaler()
        stacked = np.stack([post_test_number_of_positives, post_test_trajectory_sizes]).transpose()
        stacked_trans = scaler.fit_transform(stacked)
        combined_measure = stacked_trans.sum(axis=1).tolist()
        sum_per_set = np.sum(combined_measure) / self.number_of_folds
        folds_indices_of_post_test_trajectories = partitioning.balanced_multi_way_partition(combined_measure, sum_per_set,
                                                      len(combined_measure) // self.number_of_folds, self.number_of_folds)

        no_test_indices = [f for f in range(len(self.labels)) if f not in self.test_trajectory_indices]
        folds_indices_of_post_test_trajectories_updated = []
        for f in folds_indices_of_post_test_trajectories:
            single_fold_indices = []
            for k in f:
                single_fold_indices.append(no_test_indices[k])
            folds_indices_of_post_test_trajectories_updated.append(single_fold_indices)

        for f in folds_indices_of_post_test_trajectories_updated:
            self.window_indices_in_folds.append(np.random.permutation(np.where(np.isin(self.association, f))[0]))
        self.history['stratified cross-validation folds'] = True

    def dilate_labels(self):
        # currently only rectangular window, add other window types if needed
        # disregarding 1,2 labels - smear everything into 1 class
        window_size = self.dilation_window_size
        if not window_size % 2:
            window_size += 1
        for sequence in self.windowed_labels:
            Data.dilate_sequence(sequence, window_size)
        self.history['dilated labels'] = True

    @staticmethod
    def dilate_sequence(sequence, window_size):
        label_inds = np.nonzero(sequence)[0]
        if label_inds.size:
            i_minus = label_inds - (window_size - 1) / 2
            i_plus = label_inds + (window_size - 1) / 2
            inds_in_window = [np.arange(max(im, 0), min(ip + 1, sequence.shape[0]), dtype=int) for
                              im, ip in zip(i_minus, i_plus)]
            inds_in_window = np.unique(np.concatenate(inds_in_window))
            sequence[inds_in_window] = 1

    def detrend(self, spatial_coordinate_indices):
        model = LinearRegression()
        sequence_list = []
        for sequence in self.windowed_features:
            x = sequence[:, spatial_coordinate_indices[0]].reshape(-1, 1)
            y = sequence[:, spatial_coordinate_indices[1]]
            model.fit(x, y)
            trend = model.predict(x)

            detrended_y = y-trend
            x_zero = x-x[0]
            x_zero = np.squeeze(x_zero, axis=1)

            new_spatial_features = np.stack([x_zero, detrended_y], axis=1)
            sequence = np.insert(sequence, spatial_coordinate_indices, new_spatial_features, axis=1)

            columns_to_remove = [c[1]+1+c[0] for c in enumerate(spatial_coordinate_indices)]
            sequence_list.append(np.delete(sequence, columns_to_remove, axis=1))
        self.windowed_features = np.stack(sequence_list)
        self.history['de-trended spatial coordinates'] = spatial_coordinate_indices

    def add_aggregate_velocity_features(self, velocity_magnitude_indices):
        window_size = self.dilation_window_size
        if not window_size % 2:
            window_size += 1

        sequence_list = []
        for sequence in self.windowed_features:
            sequence_df = pd.DataFrame(sequence[:, velocity_magnitude_indices])
            rolling_mean = sequence_df.rolling(window=window_size, center=True, min_periods=1).mean()
            rolling_std = sequence_df.rolling(window=window_size, center=True, min_periods=1).std()

            sequence = np.insert(sequence, [v+1 for v in velocity_magnitude_indices], rolling_mean, axis=1)
            std_insert_indices = [k[1]+2+k[0] for k in enumerate(velocity_magnitude_indices)]
            sequence_list.append(np.insert(sequence, std_insert_indices, rolling_std, axis=1))
        self.windowed_features = np.stack(sequence_list)
        self.history['aggregated velocity features added'] = True

    def expand_label_dims(self):
        self.windowed_labels = np.expand_dims(self.windowed_labels, axis=2)
        self.history['expanded label dimension'] = True

    def augment_random_rotations(self, circular_indices, spatial_coordinate_indices, augment_factor=2):
        num_series_examples = self.windowed_features.shape[0]

        for _ in range(augment_factor):
            new_features = copy.copy(self.windowed_features[:num_series_examples, :, :])
            random_thetas_new = 360*np.random.random(num_series_examples)
            for sequence, random_theta in zip(new_features, random_thetas_new):
                for index in circular_indices:
                    sequence[:, index] = (sequence[:, index] + random_theta) % 360
                xy = sequence[:, spatial_coordinate_indices]
                c, s = np.cos(np.radians(random_theta)), np.sin(np.radians(random_theta))
                j = np.array([[c, -s], [s, c]])
                sequence[:, spatial_coordinate_indices] = np.matmul(j, xy.T).T
            self.augmentation_random_thetas.append(random_thetas_new)
            self.windowed_features = np.concatenate([self.windowed_features, new_features])
        self.windowed_labels = np.tile(self.windowed_labels, [augment_factor+1, 1, 1])
        self.augmentation_random_thetas = np.concatenate(self.augmentation_random_thetas)
        self.history['random rotation augmentation'] = True

    def augment_xy_flip(self, circular_indices, spatial_coordinate_indices):
        # flip x -> -x, y -> -y
        theta_flip = [180, 0]
        spatial_flip_ind = [0, 1]
        num_flips = len(theta_flip)
        num_series_examples = self.windowed_features.shape[0]
        for thet, spat_ind in zip(theta_flip, spatial_flip_ind):
            new_features = copy.copy(self.windowed_features[:num_series_examples])
            for sequence in new_features:
                for index in circular_indices:
                    sequence[:, index] = (thet-sequence[:, index]) % 360
                sequence[:, spatial_coordinate_indices[spat_ind]] = -sequence[:, spatial_coordinate_indices[spat_ind]]
            self.windowed_features = np.concatenate([self.windowed_features, new_features])
        self.windowed_labels = np.tile(self.windowed_labels, [num_flips+1, 1, 1])
        self.history['XY flip augmentation'] = True

    def separate_circular_features(self, circular_indices):
        sequence_list = []
        for sequence in self.windowed_features:
            rads = [np.radians(sequence[:, index].astype(float)) for index in circular_indices]
            cos_sins = np.concatenate([np.stack([np.cos(r), np.sin(r)], axis=1) for r in rads], axis=1)

            doubled_indices = list(itertools.chain.from_iterable([[c, c] for c in circular_indices]))
            sequence = np.insert(sequence, doubled_indices, cos_sins, axis=1)

            columns_to_remove = [c[1] + 2 + c[0] * 2 for c in enumerate(circular_indices)]
            sequence_list.append(np.delete(sequence, columns_to_remove, axis=1))
        self.windowed_features = np.stack(sequence_list)
        self.history[r'separated circular features into cos/sin'] = True


def divisible_window_size(orig_size, divider):
    return orig_size + (divider - orig_size % divider)



