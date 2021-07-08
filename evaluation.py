import numpy as np


def final_sequence_from_labeled_windows(labels, stride):
    # todo: test
    labels[labels == 0] = -1
    num_windows = labels.shape[0]-1
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


def evaluate_sequence(predicted_full_sequence, actual_full_sequence):

    return