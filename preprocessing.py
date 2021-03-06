import pandas as pd
import numpy as np
import math
import itertools
import pickle
import copy
import partitioning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import random

# import timeit
# starttime = timeit.default_timer()
# print("The start time is :",starttime)
# test()
# print("The time difference is :", timeit.default_timer() - starttime)

# region Globals
MARKING_TIME = 0.1  # typical marking time in sec
fps = 50
# endregion Globals

def get_set_randomality():
    filename = 'pickled_random_state.pic'
    try:
        with open(filename, 'rb') as f:
            prng = pickle.load(f)
        state = prng.get_state()
        np.random.set_state(state)
    except FileNotFoundError:
        prng = np.random.RandomState()
        with open(filename, 'wb') as f:
            pickle.dump(prng, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        raise error


def transform(features, indices_to_transform, transformer):
    for i in indices_to_transform:
        stscaler = transformer()
        only_feature_list = [p[:, i] for p in features]
        only_feature_array = np.concatenate(only_feature_list)
        stscaler.fit(only_feature_array.reshape(-1, 1))
        for sequence in features:
            sequence[:, i] = np.squeeze(stscaler.transform(sequence[:, i].reshape(-1, 1)))


def separate_any_positive_examples(features, labels):
    num_positives = np.array([sum(l) for l in labels])
    where_positives = np.nonzero(num_positives)[0]
    positive_labels = [labels[w] for w in where_positives]
    positive_features = [features[w] for w in where_positives]

    where_negatives = list(set(list(range(len(num_positives)))) - set(where_positives))
    negative_labels = [labels[w] for w in where_negatives]
    negative_features = [features[w] for w in where_negatives]
    return positive_labels, positive_features, where_positives, negative_labels, negative_features


def divisible_window_size(orig_size, divider):
    return orig_size + (divider - orig_size % divider)


def cut_array_into_windows(features, labels, window_size, stride):
    # Cut data and labels into overlapping windows of size window_size, with stride stride.
    # Zero-pad samples with size<window_size

    label_list = []
    feature_list = []
    traj_index_list = []
    count = 0
    for feature_sequence, label_sequence in zip(features, labels):
        length = feature_sequence.shape[0]
        last_full_window_ind = max(math.floor((length-window_size)/stride+1), 1)
        seq_feature_list = []
        seq_label_list = []
        for i in range(0, length, stride):
            seq_feature_list.append(feature_sequence[i: i + window_size, :])
            seq_label_list.append(label_sequence[i: i + window_size])

        seq_feature_list = seq_feature_list[:last_full_window_ind]  # remove last part to make sure sizes are the same
        seq_label_list = seq_label_list[:last_full_window_ind]

        if last_full_window_ind == 1:  # Zero-pad samples with size<window_size
            seq_feature_list[0] = np.pad(seq_feature_list[0], ((0, window_size - seq_feature_list[0].shape[0]), (0, 0)))
            seq_label_list[0] = np.pad(seq_label_list[0], (0, window_size - seq_label_list[0].shape[0]))

        feature_list.extend(seq_feature_list)
        label_list.extend(seq_label_list)
        traj_index_list.extend([count] * len(seq_feature_list))
        count += 1

    return np.stack(feature_list), np.stack(label_list), np.array(traj_index_list)


def get_test_data(features, labels, trajectory_association, percent_test, original_trajectory_sizes):
    total_test_trajectory_length = round(percent_test*sum(original_trajectory_sizes)/100)
    max_length = round(percent_test*len(original_trajectory_sizes)/100)
    subset, indices_of_subset_in_original = partitioning.find_approximate_subset(
        original_trajectory_sizes, list(range(len(original_trajectory_sizes))), total_test_trajectory_length, max_length)
    indices_of_subset = np.where(np.isin(trajectory_association, indices_of_subset_in_original))[0]

    test = {'features': features[indices_of_subset], 'labels': labels[indices_of_subset], 'association':
            trajectory_association[indices_of_subset]}
    trajectory_association = np.delete(trajectory_association, indices_of_subset)
    features = np.delete(features, indices_of_subset, axis=0)
    labels = np.delete(labels, indices_of_subset, axis=0)
    return test, trajectory_association, features, labels


def stratify_training_data(features, labels, trajectory_association, original_trajectory_sizes,
                           original_number_of_positives, number_of_folds):
    scaler = MinMaxScaler()
    stacked = np.stack([original_number_of_positives, original_trajectory_sizes]).transpose()
    scaler.fit(stacked)
    stacked_trans = scaler.fit_transform(stacked)
    combined_measure = stacked_trans.sum(axis=1).tolist()
    sum_per_set = np.sum(combined_measure) / number_of_folds
    folds_indices_of_orig_trajectories = partitioning.balanced_multi_way_partition(combined_measure, sum_per_set,
                                           len(combined_measure) // number_of_folds, number_of_folds)

    window_indices_in_folds = []
    for f in folds_indices_of_orig_trajectories:
        window_indices_in_folds.append(np.random.permutation(np.where(np.isin(trajectory_association, f))[0]))

    training = {'features': features, 'labels': labels, 'association': trajectory_association}
    return training,  window_indices_in_folds




def create_validation_set(training, folds_indices, fold_num=0):
    validation = {}
    validation['features'] = training['features'][folds_indices[fold_num]]
    validation['labels'] = training['labels'][folds_indices[fold_num]]
    validation['association'] = training['association'][folds_indices[fold_num]]

    training['features'] = np.delete(training['features'], folds_indices[fold_num], axis=0)
    training['labels'] = np.delete(training['labels'], folds_indices[fold_num], axis=0)
    training['association'] = np.delete(training['association'], folds_indices[fold_num])
    return validation

def take_only_positive_windows(training):
     num_positives = np.sum(training['labels'], axis=1)
     where_positives = np.nonzero(num_positives)[0]
     where_negatives = list(set(list(range(len(num_positives)))) - set(where_positives))
     training['labels'] = np.delete(training['labels'], where_negatives, axis=0)
     training['features'] = np.delete(training['features'], where_negatives, axis=0)

def dilate_labels(training, window_size):
    # currently only rectangular window, add other window types if needed
    # disregarding 1,2 labels - smear everything into 1 class

    if not window_size % 2:
        window_size += 1
    for sequence in training['labels']:
        dilate_sequence(sequence, window_size)


def dilate_sequence(sequence, window_size):
    label_inds = np.nonzero(sequence)[0]
    if label_inds.size:
        i_minus = label_inds - (window_size - 1) / 2
        i_plus = label_inds + (window_size - 1) / 2
        inds_in_window = [np.arange(max(im, 0), min(ip + 1, sequence.shape[0]), dtype=int) for
                          im, ip in zip(i_minus, i_plus)]
        inds_in_window = np.unique(np.concatenate(inds_in_window))
        sequence[inds_in_window] = 1

def augment_random_rotations(training, circular_indices, spatial_coordinate_indices, augment_factor=2):
    random_thetas = []
    num_series_examples = training['features'].shape[0]

    for _ in range(augment_factor):
        new_features = copy.copy(training['features'][:num_series_examples, :, :])
        random_thetas_new = 360*np.random.random(num_series_examples)
        for sequence, random_theta in zip(new_features, random_thetas_new):
            for index in circular_indices:
                sequence[:, index] = (sequence[:, index] + random_theta) % 360
            xy = sequence[:, spatial_coordinate_indices]
            c, s = np.cos(np.radians(random_theta)), np.sin(np.radians(random_theta))
            j = np.array([[c, -s], [s, c]])
            sequence[:, spatial_coordinate_indices] = np.matmul(j, xy.T).T
        random_thetas.append(random_thetas_new)
        training['features'] = np.concatenate([training['features'], new_features])
    training['labels'] = np.tile(training['labels'], [augment_factor+1, 1, 1])
    random_thetas = np.concatenate(random_thetas)
    return random_thetas


def augment_xy_flip(training, circular_indices, spatial_coordinate_indices):
    # flip x -> -x, y -> -y
    theta_flip = [180, 0]
    spatial_flip_ind = [0, 1]
    num_flips = len(theta_flip)
    num_series_examples = training['features'].shape[0]
    for thet, spat_ind in zip(theta_flip, spatial_flip_ind):
        new_features = copy.copy(training['features'][:num_series_examples])
        for sequence in new_features:
            for index in circular_indices:
                sequence[:, index] = (thet-sequence[:, index]) % 360
            sequence[:, spatial_coordinate_indices[spat_ind]] = -sequence[:, spatial_coordinate_indices[spat_ind]]
        training['features'] = np.concatenate([training['features'], new_features])
    training['labels'] = np.tile(training['labels'], [num_flips+1, 1, 1])


def detrend(data, spatial_coordinate_indices):
    model = LinearRegression()
    sequence_list = []
    for sequence in data['features']:
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
    data['features'] = np.stack(sequence_list)


def separate_circular_features(data, circular_indices):
    sequence_list = []
    for sequence in data['features']:
        rads = [np.radians(sequence[:, index].astype(float)) for index in circular_indices]
        cos_sins = np.concatenate([np.stack([np.cos(r), np.sin(r)], axis=1) for r in rads], axis=1)

        doubled_indices = list(itertools.chain.from_iterable([[c, c] for c in circular_indices]))
        sequence = np.insert(sequence, doubled_indices, cos_sins, axis=1)

        columns_to_remove = [c[1]+2+c[0]*2 for c in enumerate(circular_indices)]
        sequence_list.append(np.delete(sequence, columns_to_remove, axis=1))
    data['features'] = np.stack(sequence_list)


def add_aggregate_velocity_features(data, velocity_magnitude_indices):
    window_size = math.ceil(MARKING_TIME * fps)
    if not window_size % 2:
        window_size += 1

    sequence_list = []
    for sequence in data['features']:
        sequence_df = pd.DataFrame(sequence[:, velocity_magnitude_indices])
        rolling_mean = sequence_df.rolling(window=window_size, center=True, min_periods=1).mean()
        rolling_std = sequence_df.rolling(window=window_size, center=True, min_periods=1).std()

        sequence = np.insert(sequence, [v+1 for v in velocity_magnitude_indices], rolling_mean, axis=1)
        std_insert_indices = [k[1]+2+k[0] for k in enumerate(velocity_magnitude_indices)]
        sequence_list.append(np.insert(sequence, std_insert_indices, rolling_std, axis=1))
    data['features'] = np.stack(sequence_list)


def show_shapes(training, validation, test):
    for data, dataname in zip([training, validation, test], ('training', 'validation', 'test')):
        print(dataname + ':')
        print("Expected: (num_samples, timesteps, features)")
        print("Sequences: {}".format(data['features'].shape))
        print("Targets:   {}".format(data['labels'].shape))
        print('\n')

