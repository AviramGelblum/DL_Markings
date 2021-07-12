import numpy as np
import preprocessing as pp


def label_chunks(labels, association_values):
    unique_vals = np.unique(association_values)
    for val in unique_vals:
        yield labels[np.where(association_values == val), :]

def final_sequence_from_labeled_windows(labels, stride):
    # todo: test
    labels[labels == 0] = -1
    num_windows = labels.shape[0]
    window_size = labels.shape[1]
    original_sequence_size = stride*(num_windows-1)+window_size
    padded_sequence_list = []
    for starting_index, sequence in zip(range(0, num_windows*stride, stride), labels):
        padded_sequence_list.append(np.pad(sequence, (starting_index, original_sequence_size-(
                                           window_size+starting_index))))
    padded_sequence = np.stack(padded_sequence_list, axis=0)
    final = np.add.reduce(padded_sequence, axis=0)
    final[final >= 0] = 1
    final[final < 0] = 0
    return final


def evaluate_sequence(predicted, actual, tolerance: int):
    dilated_predicted = pp.dilate_sequence(predicted, tolerance)
    dilated_actual = pp.dilate_sequence(actual, tolerance)

    # true positive - any overlap between dilated predicted sequence and non-dilated actual markings
    TP = np.bitwise_and(dilated_predicted, actual).sum()

    # false positive - any non-dilated predicted sequence positives not overlapping with dilated actual markings
    FP = np.where(predicted - dilated_actual == 1)[0].shape[0]

    # false negative - any non-dilated actual markings not overlapping with dilated predicted:
    FN = np.where(actual - dilated_predicted == 1)[0].shape[0]

    return TP, FP, FN


def get_metrics(TP, FP, FN):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*(precision*recall)/(precision+recall)
    return precision, recall, F1