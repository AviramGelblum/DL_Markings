import numpy as np
import math
import itertools
import pickle
import copy
import partitioning

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


def separate_any_positive_examples(features, labels):
    num_positives = np.array([sum(l) for l in labels])
    where_positives = np.nonzero(num_positives)[0]
    positive_labels = [labels[w] for w in where_positives]
    positive_features = [features[w] for w in where_positives]

    where_negatives = list(set(list(range(len(num_positives)))) - set(where_positives))
    negative_labels = [labels[w] for w in where_negatives]
    negative_features = [features[w] for w in where_negatives]
    return positive_labels, positive_features, where_positives, negative_labels, negative_features


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
        # stack_list.append(np.stack(sequence_list, axis=0))

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
    # indices_of_subset_in_original = np.where(np.isin(original_trajectory_sizes, subset))[0]
    indices_of_subset = np.where(np.isin(trajectory_association, indices_of_subset_in_original))[0]

    test = {'features': features[indices_of_subset,:,:], 'labels': labels[indices_of_subset,:], 'association':
            trajectory_association[indices_of_subset]}
    trajectory_association = np.delete(trajectory_association, indices_of_subset)
    features = np.delete(features, indices_of_subset, axis=0)
    labels = np.delete(labels, indices_of_subset, axis=0)
    return test, trajectory_association, features, labels


def stratify_training_data(features, labels, trajectory_association, number_of_folds):
    num_positives = np.sum(labels, axis=1)
    where_positives = np.nonzero(num_positives)[0]
    only_positives = list(num_positives[where_positives])
    sum_per_set = round(np.sum(only_positives)/number_of_folds)
    num_positive_sequences = only_positives.__len__()
    folds_indices_in_positives = partitioning.balanced_multi_way_partition(only_positives, sum_per_set,
                                                         round(num_positive_sequences/number_of_folds), number_of_folds)

    folds_indices = [where_positives[f] for f in folds_indices_in_positives]

    where_negatives = list(set(list(range(len(num_positives)))) - set(where_positives))
    num_negative_sequences = len(where_negatives)
    num_negatives_in_fold = int(np.floor(num_negative_sequences/number_of_folds))
    range_negatives = list(range(0, num_negatives_in_fold * (number_of_folds+1), num_negatives_in_fold))
    folds_indices = [np.random.permutation(np.append(fold, where_negatives[k:k2])) for fold, k, k2
                     in zip(folds_indices, range_negatives[:-1], range_negatives[1:])]
    training = {'features': features, 'labels': labels, 'association': trajectory_association}
    return training,  folds_indices  # if we need folds


def create_validation_set(training, folds_indices, fold_num=0):
    validation = {}
    validation['features'] = training['features'][folds_indices[fold_num], :, :]
    validation['labels'] = training['labels'][folds_indices[fold_num], :]
    validation['association'] = training['association'][folds_indices[fold_num]]

    training['features'] = np.delete(training['features'], folds_indices[fold_num], axis=0)
    training['labels'] = np.delete(training['labels'], folds_indices[fold_num], axis=0)
    training['association'] = np.delete(training['association'], folds_indices[fold_num])
    return validation


def smear_labels(training):
    # currently only rectangular window, add other window types if needed
    # disregarding 1,2 labels - smear everything into 1 class
    window_size = math.ceil(MARKING_TIME*fps)
    if not window_size % 2:
        window_size += 1
    for sequence in training['labels']:
        label_inds = np.nonzero(sequence)[0]
        if label_inds.size:
            i_minus = label_inds-(window_size-1)/2
            i_plus = label_inds+(window_size-1)/2
            inds_in_window = [np.arange(max(im, 0), min(ip+1, sequence.shape[0]), dtype=int) for
                              im, ip in zip(i_minus, i_plus)]
            inds_in_window = np.unique(np.concatenate(inds_in_window))
            sequence[inds_in_window] = 1


def augment_random_rotations(training, circular_indices, spatial_coordinate_indices, augment_coefficient=2):
    random_thetas = []
    orig_features_len = training['features'].shape[0]

    for aug_ind in range(augment_coefficient):
        new_features = copy.deepcopy(training['features'][:orig_features_len, :, :])
        random_thetas_new = 360*np.random.random(orig_features_len)
        for sequence, random_theta in zip(new_features, random_thetas_new):
            for index in circular_indices:
                sequence[:, index] = (sequence[:, index] + random_theta) % 360
            xy = sequence[:, spatial_coordinate_indices]
            c, s = np.cos(np.radians(random_theta)), np.sin(np.radians(random_theta))
            j = np.array([[c, -s], [s, c]])
            sequence[:, spatial_coordinate_indices] = np.matmul(j, xy.T).T
        random_thetas.append(random_thetas_new)
        training['features'] = np.concatenate([training['features'], new_features])
    training['labels'] = np.tile(training['labels'], [augment_coefficient+1, 1])
    random_thetas = np.concatenate(random_thetas)
    return random_thetas

def augment_mirroring(training, circular_indices, spatial_coordinate_indices):
    # TODO: augmentation - mirroring


def reduce_space(data, spatial_coordinate_indices):
    # TODO: feature engineering - x,y - subtract linear fit from data to get a one dimensional value

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


def add_aggregate_velocity_features(data):
    # TODO: feature engineering - mean and std of velocity, scale of marking (200 ms). maybe other measures (kurtosis,
    #  skew)?

