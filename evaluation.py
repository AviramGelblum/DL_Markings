import numpy as np
import preprocessing as pp
import keras.callbacks as Kcallbacks
import math
import copy
import model
import matplotlib.pyplot as plt

def label_chunks(labels, association_values):
    unique_vals = np.unique(association_values)
    for val in unique_vals:
        yield labels[np.where(association_values == val)]


def final_sequence_from_labeled_windows(labels, stride):
    labels = labels.astype(int)
    labels[labels == 0] = -1
    num_windows = labels.shape[0]
    window_size = labels.shape[1]
    original_sequence_size = stride*(num_windows-1)+window_size
    padded_sequence_list = []
    for starting_index, sequence in zip(range(0, num_windows*stride, stride), labels):
        padded_sequence_list.append(np.pad(sequence,
                                    ((starting_index, original_sequence_size-(window_size+starting_index)), (0, 0))))
    padded_sequence = np.stack(padded_sequence_list, axis=0)
    final = np.add.reduce(padded_sequence, axis=0)
    final[final >= 0] = 1
    final[final < 0] = 0
    return final


def evaluate_sequence(predicted, actual, tolerance: int):
    dilated_predicted = copy.copy(predicted)
    dilated_actual = copy.copy(actual)
    pp.dilate_sequence(dilated_predicted, tolerance)
    pp.dilate_sequence(dilated_actual, tolerance)

    # true positive - any overlap between dilated predicted sequence and non-dilated actual markings
    TP = np.logical_and(dilated_predicted, actual).sum()

    # false positive - any non-dilated predicted sequence positives not overlapping with dilated actual markings
    FP = np.where(predicted - dilated_actual == 1)[0].shape[0]

    # false negative - any non-dilated actual markings not overlapping with dilated predicted:
    FN = np.where(actual - dilated_predicted == 1)[0].shape[0]

    return TP, FP, FN


def simple_metrics(TP, FP, FN):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*(precision*recall)/(precision+recall)
    return precision, recall, f1


def multi_thresh_metrics(predicted_probabilities, validation, validation_full_targets, stride, dilation_window_size):
    precision = []
    recall = []
    f1 = []
    for thresh in np.linspace(0, 1, 101):
        TP = FP = FN = 0
        predicted_labels = predicted_probabilities > thresh
        for chunk, actual_full_sequence in zip(label_chunks(predicted_labels, validation['association']),
                                               validation_full_targets):
            predicted_full_sequence = final_sequence_from_labeled_windows(chunk, stride).astype(float)
            actual_full_sequence = actual_full_sequence.astype(float).reshape(-1,1)
            if actual_full_sequence.shape[0] > predicted_full_sequence.shape[0]:
                actual_full_sequence = actual_full_sequence[:predicted_full_sequence.shape[0]]
            elif actual_full_sequence.shape[0] < predicted_full_sequence.shape[0]:
                predicted_full_sequence = predicted_full_sequence[:actual_full_sequence.shape[0]]
            TP_seq, FP_seq, FN_seq = evaluate_sequence(predicted_full_sequence, actual_full_sequence,
                                                       dilation_window_size)
            TP += TP_seq
            FP += FP_seq
            FN += FN_seq
        precision_out, recall_out, f1_out = simple_metrics(TP, FP, FN)
        precision.append(precision_out)
        recall.append(recall_out)
        f1.append(f1_out)
    return precision, recall, f1

def draw_PR_curve(precision,recall):
    plt.figure('Precision vs Recall curve')
    plt.plot(recall, precision, color='red', marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show(block=False)

class DilatedOnTrainingEvaluation(Kcallbacks.Callback):
    def __init__(self, model, validation, tolerance=math.ceil(pp.MARKING_TIME * pp.fps)*2):
        self.model = model
        self.val_inputs = validation['features']
        self.val_targets = validation['labels']
        self.tolerance = tolerance

    def on_epoch_end(self, epoch, logs=None):
        all_predicted = self.model.predict(self.val_inputs)
        TP = FP = FN = 0
        probability_threshold = model.training_parameters['probability_threshold']
        for predicted_probabilities, target in zip(all_predicted, self.val_targets):
            predicted = (predicted_probabilities > probability_threshold).astype(float)
            TP_seq, FP_seq, FN_seq = evaluate_sequence(predicted, target, self.tolerance)
            TP += TP_seq
            FP += FP_seq
            FN += FN_seq
        precision, recall, f1 = simple_metrics(TP, FP, FN)
        print('val_special_precision:', precision)
        print('val_special_recall:', recall)
        print("val_special_f1:", f1)

