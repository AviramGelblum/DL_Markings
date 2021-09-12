import numpy as np
import Data
import keras.callbacks as Kcallbacks
import matplotlib.pyplot as plt
import tabulate
from Pipeline import Iprocessable


class Evaluation(Iprocessable):
    def __init__(self, runner, thresholds=np.linspace(0, 1, 101)):
        self.validation = runner.model.validation_data
        self.validation_fold_index = runner.model.validation_fold_index
        self.predicted_probabilities = runner.model.validation_predicted_probabilities

        self.stride = runner.model_parameters['stride']
        self.sample_window_size = runner.model_parameters['sample_window_size']
        self.prediction_window_size = runner.training_parameters['prediction_window_size']
        self.probability_threshold = runner.training_parameters['probability_threshold']

        self.metrics = None
        self.thresholds = thresholds

        self.history = runner.history

    def verify_type(self, instance_types):
        return True

    @staticmethod
    def evaluate_sequence(predicted, actual, tolerance: int):
        dilated_predicted = np.copy(predicted)
        dilated_actual = np.copy(actual)
        Data.Data.dilate_sequence(dilated_predicted, tolerance)
        Data.Data.dilate_sequence(dilated_actual, tolerance)

        # true positive - any overlap between dilated predicted sequence and non-dilated actual markings
        TP = int(np.logical_and(dilated_predicted, actual).sum())

        # false positive - any non-dilated predicted sequence positives not overlapping with dilated actual markings
        FP = np.where(predicted - dilated_actual == 1)[0].shape[0]

        # false negative - any non-dilated actual markings not overlapping with dilated predicted:
        FN = np.where(actual - dilated_predicted == 1)[0].shape[0]

        return TP, FP, FN

    @staticmethod
    def simple_metrics(TP, FP, FN):
        if TP == 0 and FP == 0:
            precision = 1
        else:
            precision = TP/(TP+FP)

        if TP == 0 and FN == 0:
            recall = 1
        else:
            recall = TP/(TP+FN)

        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2*(precision*recall)/(precision+recall)

        return precision, recall, f1

    def multi_thresh_metrics(self):
        MTMCalculator = MultiThresholdMetricsCalculator(self)
        self.metrics = MTMCalculator.run()

    def get_predicted_markings(self,  threshold=None, threshold_type='probability'):
        accepted_strings = {'recall', 'precision', 'probability'}

        if isinstance(threshold_type, str):
            if threshold_type not in accepted_strings:
                raise ValueError('threshold_type must have one of the following values: ' + str(accepted_strings))
        else:
            raise TypeError('threshold_type must be a string.')

        if threshold_type == 'probability':
            if threshold is None:
                threshold = self.probability_threshold
        else:
            if threshold is None:
                raise ValueError('If threshold_type is ' + threshold_type + ', threshold must be specified.')
            threshold = self.thresholds[(np.abs(np.array(self.metrics[threshold_type]) - threshold)).argmin()]

        PMGetter = PredictedMarkingsGetter(self, threshold)
        return PMGetter.run()

    def draw_precision_recall_curve(self, title=''):
        plt.figure('Precision vs Recall curve')  # + str(fig_num))
        plt.plot(self.metrics['recall'], self.metrics['precision'],
                 marker='o', label='fold number ' + str(self.validation_fold_index+1))
        plt.title(title)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show(block=False)


def print_cross_validation_results(evaluations_list, threshold=None):
    if threshold is None:
        threshold = evaluations_list[0].probability_threshold
    thresh_ind = (np.abs(evaluations_list[0].thresholds - threshold)).argmin()
    results_matrix = [[ev[0] + 1,
                       ev[1].metrics['precision'][thresh_ind],
                       ev[1].metrics['recall'][thresh_ind],
                       ev[1].metrics['f1'][thresh_ind]]
                      for ev in enumerate(evaluations_list)]
    print('\nCross-validation results at probability threshold = ' + str(threshold) + '\n')
    print(tabulate.tabulate(results_matrix, headers=["Fold", "Precision", "Recall", "F1"], tablefmt='grid'))


class ChunkProcess:
    def __init__(self, evaluation_object):
        self.eval_obj = evaluation_object
        self.predicted_labels = None

    def _label_chunks(self):
        unique_vals = np.unique(self.eval_obj.validation.association)
        for val in unique_vals:
            labels_out = self.predicted_labels[np.where(self.eval_obj.validation.association == val)]
            labels_out = labels_out[np.argsort(self.eval_obj.validation.window_indices_in_folds[self.eval_obj.validation_fold_index]
                                               [np.where(self.eval_obj.validation.association == val)])]
            yield labels_out, val

    def _final_sequence_from_labeled_windows(self, labels):
        labels = labels.astype(int)
        labels[labels == 0] = -1
        num_windows = labels.shape[0]
        original_sequence_size = self.eval_obj.stride * (num_windows - 1) + self.eval_obj.sample_window_size
        padded_sequence_list = []
        for starting_index, sequence in zip(range(0, num_windows * self.eval_obj.stride, self.eval_obj.stride),
                                            labels):
            padded_sequence_list.append(np.pad(sequence,
                                               ((starting_index, original_sequence_size - (self.eval_obj.sample_window_size + starting_index)),
                                                (0, 0))))
        padded_sequence = np.stack(padded_sequence_list, axis=0)
        final = np.add.reduce(padded_sequence, axis=0)
        final[final >= 0] = 1
        final[final < 0] = 0
        return final

    def _get_full_sequences(self, chunk, actual_full_sequence):
        predicted_full_sequence = self._final_sequence_from_labeled_windows(chunk).astype(float)
        if actual_full_sequence.shape[0] > predicted_full_sequence.shape[0]:
            actual_full_sequence = actual_full_sequence[:predicted_full_sequence.shape[0]]
        elif actual_full_sequence.shape[0] < predicted_full_sequence.shape[0]:
            predicted_full_sequence = predicted_full_sequence[:actual_full_sequence.shape[0]]
        return predicted_full_sequence, actual_full_sequence

    def _chunk_loop(self):
        lc_gen = self._label_chunks()
        for chunk, actual_full_sequence in zip(lc_gen, self.eval_obj.validation.labels):
            actual_full_sequence = actual_full_sequence.astype(float).reshape(-1, 1)
            predicted_full_sequence, actual_full_sequence = self._get_full_sequences(chunk[0], actual_full_sequence)
            args = {'predicted_full_sequence': predicted_full_sequence,
                    'actual_full_sequence': actual_full_sequence,
                    'chunk': chunk}
            self._chunk_internal(args)

    def _chunk_internal(self, args):
        pass


class PredictedMarkingsGetter(ChunkProcess):
    def __init__(self, evaluation_object, threshold):
        super().__init__(evaluation_object)
        self.predicted_labels = self.eval_obj.predicted_probabilities > threshold
        self.predicted_full_sequences = {}

    def run(self):
        super()._chunk_loop()
        return self.predicted_full_sequences

    def _chunk_internal(self, args):
        dict_key = str(args['chunk'][1])
        self.predicted_full_sequences[dict_key] = args['predicted_full_sequence']


class ThresholdChunkProcess(ChunkProcess):
    def __init__(self, evaluation_object):
        super().__init__(evaluation_object)
        self.thresholds = evaluation_object.thresholds

    def _threshold_loop(self):
        for threshold in self.thresholds:
            self._threshold_internal_initial()
            self.predicted_labels = self.eval_obj.predicted_probabilities > threshold
            self._chunk_loop()
            self._threshold_internal_final()

    def _threshold_internal_initial(self):
        pass

    def _threshold_internal_final(self):
        pass


class MultiThresholdMetricsCalculator(ThresholdChunkProcess):
    def __init__(self, evaluation_object):
        super().__init__(evaluation_object)
        self.basic_metrics = {'TP': 0, 'FP': 0, 'FN': 0}
        self.metrics = {'precision': [], 'recall': [], 'f1': []}

    def run(self):
        super()._threshold_loop()
        return self.metrics

    def _threshold_internal_initial(self):
        self.basic_metrics = {'TP': 0, 'FP': 0, 'FN': 0}

    def _threshold_internal_final(self):
        precision_out, recall_out, f1_out = Evaluation.simple_metrics(self.basic_metrics['TP'],
                                                                      self.basic_metrics['FP'],
                                                                      self.basic_metrics['FN'])
        self.metrics['precision'].append(precision_out)
        self.metrics['recall'].append(recall_out)
        self.metrics['f1'].append(f1_out)

    def _chunk_internal(self, args):
        TP_seq, FP_seq, FN_seq = Evaluation.evaluate_sequence(args['predicted_full_sequence'],
                                args['actual_full_sequence'], self.eval_obj.prediction_window_size)
        self.basic_metrics['TP'] += TP_seq
        self.basic_metrics['FP'] += FP_seq
        self.basic_metrics['FN'] += FN_seq


class DilatedOnTrainingEvaluation(Kcallbacks.Callback):
    def __init__(self, model, validation: Data.Data, tolerance=7, probability_threshold=0.25):
        self.model = model
        self.val_inputs = validation.windowed_features
        self.val_targets = validation.windowed_labels
        self.tolerance = tolerance
        self.threshold = probability_threshold

    def on_epoch_end(self, epoch, logs=None):
        all_predicted = self.model.predict(self.val_inputs)
        TP = FP = FN = 0
        for predicted_probabilities, target in zip(all_predicted, self.val_targets):
            predicted = (predicted_probabilities > self.threshold).astype(float)
            TP_seq, FP_seq, FN_seq = Evaluation.evaluate_sequence(predicted, target, self.tolerance)
            TP += TP_seq
            FP += FP_seq
            FN += FN_seq
        precision, recall, f1 = Evaluation.simple_metrics(TP, FP, FN)
        print('val_special_precision:', precision)
        print('val_special_recall:', recall)
        print("val_special_f1:", f1)

