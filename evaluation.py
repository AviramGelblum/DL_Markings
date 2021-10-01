from __future__ import annotations

from Data import Data
from Pipeline import IProcessable

from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import tabulate
from typing import Optional, TYPE_CHECKING

import keras.callbacks as K_callbacks

if TYPE_CHECKING:
    # Always False, allows compile-time type checking of variables of classes whose importing would raise run-time
    # circular import errors or are simply not used during run-time. Could simply use if False here, but this way
    # provides a better understanding of what this does and Pycharm doesn't raise a warning regarding unreachable code.
    from Runner import Runner


# region Evaluation Class
class Evaluation(IProcessable):
    # noinspection SpellCheckingInspection
    """
    Class for evaluation of Keras deep learning model performance, implementing the IProcessable interface allowing
    instances to be processed in a Pipeline.
    """

    # region Constructor
    def __init__(self, runner: Runner, thresholds: np.array = np.linspace(0, 1, 101)):
        """
        Basic constructor method for the Evaluation class.
        :param runner: Parent Runner object
        :param thresholds: Numpy array containing the probability thresholds to use when evaluating the performance of
        the model in terms of precision, recall, and F1.
        """
        # Validation Data Attributes
        # See Data.Data.__init__ method [region Constructors] for explanation of each attribute's contents
        self.validation_association = runner.model.validation_data.association
        self.validation_labels = runner.model.validation_data.labels
        self.validation_names = runner.model.validation_data.names
        self.validation_time = runner.model.validation_data.time
        self.validation_window_indices_in_folds = runner.model.validation_data.window_indices_in_folds
        self.validation_fold_index = runner.model.validation_fold_index  # Data split configuration used in the current model

        # Prediction Parameters Attributes
        # See Runner.Runner [region Basic Parameters and Defaults] for explanation of each parameter
        self.prediction_window_size = runner.training_parameters['prediction_window_size']
        self.probability_threshold = runner.training_parameters['probability_threshold']
        # Numpy array defining a list of probability thresholds to use when evaluating the performance of the model in
        # terms of precision, recall and F1.
        self.thresholds = thresholds

        # Prediction Results Attributes
        self.predicted_probabilities = runner.model.validation_predicted_probabilities  # Post-training model predictions on validation data
        self.metrics = None  # Placeholder for dictionary containing precision, recall and F1 metrics calculated for different probability thresholds

        # Model Parameter Attributes
        # See Runner.Runner [region Basic Parameters and Defaults] for explanation of each parameter
        self.stride = runner.data_parameters['stride']
        self.sample_window_size = runner.data_parameters['sample_window_size']

        # Other Attributes
        self.history = runner.history  # Dictionary containing information about previous processing
    # endregion

    # region Overriding Methods
    def verify_type(self, instance_types: None):
        """
        Verify that the current object type is in a list of accepted instance-types. This method is called during
        processing of a Pipeline object. Some methods in the pipe should process only objects of a certain type.
        :param instance_types: List of enum types, if relevant or None if irrelevant
        :return: bool flag determining if object's type is in the accepted list
        """
        return True  # No type verification for Evaluation objects - always process
    # endregion

    # region Static Methods
    # noinspection PyPep8Naming
    @staticmethod
    def evaluate_sequence(predicted: np.array, actual, tolerance: int):
        """
        Calculate number of true positive, false positive and false negative predictions in a single label sequence.
        :param predicted: Numpy array containing a sequence of binary labels predicted by the model, given some
        probability threshold.
        :param actual: Numpy array containing the actual sequence wherein marking events are marked as positives.
        :param tolerance: Window size allowing some tolerance built into the definition of what a correct prediction is,
         as explained in the comments within the method.
        :return: TP - total number of true positives in the sequence.
                 FP - total number of false positives in the sequence.
                 FN - total number of false negatives in the sequence.
        """

        # dilate the predicted and actual sequences, for a tolerant calculation of the simple metrics which allows
        # positive predictions to be off by a certain distance from an actual marking and still considered to be true.
        dilated_predicted = np.copy(predicted)
        dilated_actual = np.copy(actual)
        Data.dilate_sequence(dilated_predicted, tolerance)
        Data.dilate_sequence(dilated_actual, tolerance)

        # True Positive - Any overlap between dilated predicted sequence and non-dilated actual markings
        TP = int(np.logical_and(dilated_predicted, actual).sum())

        # False Positive - Any non-dilated predicted sequence positives not overlapping with dilated actual markings
        FP = np.where(predicted - dilated_actual == 1)[0].shape[0]

        # False Negative - Any non-dilated actual markings not overlapping with dilated predicted:
        FN = np.where(actual - dilated_predicted == 1)[0].shape[0]

        return TP, FP, FN

    @staticmethod
    def simple_metrics(TP, FP, FN):
        """
        Calculate precision, recall and F1 from number of true positives, false positives and false negatives.
        :param TP: Number of true positives.
        :param FP: Number of false positives.
        :param FN: Number of false negatives.
        :return: Calculated precision, recall and F1 metrics.
        """
        # Calculate precision
        if TP == 0 and FP == 0:  # degenerate case of no predicted positives, recall = 0 if any positives actually exist
            precision = 1
        else:
            precision = TP / (TP + FP)

        # Calculate recall
        if TP == 0 and FN == 0:  # degenerate case of no actual positives, precision = 0 if any positives were predicted
            recall = 1
        else:
            recall = TP / (TP + FN)

        # Calculate F1
        if precision == 0 and recall == 0:  # no predicted positives were correct, and positives actually exist
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1
    # endregion

    # region Other Methods
    def multi_thresh_metrics(self):
        """
         Calculate precision, recall and F1 metrics at different probability thresholds.
        """
        # noinspection PyPep8Naming
        MTMCalculator = MultiThresholdMetricsCalculator(self)
        self.metrics = MTMCalculator.run()  # store calculation results dictionary in self.metrics

    def get_predicted_markings(self, threshold: Optional[float] = None, threshold_type: str = 'probability'):
        """
        Obtain predicted label sequences for a given probability/recall/precision threshold
        :param threshold: threshold used for the creation of the binary label sequences from the predicted
        probability sequences
        :param threshold_type: type of threshold (probability, recall or precision)
        :return: List of PredictedPositives struct-like objects, containing video number, ant number and marking events frame numbers.
        """
        # Validate threshold_type string
        accepted_strings = {'recall', 'precision', 'probability'}
        if isinstance(threshold_type, str):
            if threshold_type not in accepted_strings:
                raise ValueError('threshold_type must have one of the following values: ' + str(accepted_strings))
        else:
            raise TypeError('threshold_type must be a string.')

        # Get probability threshold from input threshold
        if threshold_type == 'probability':
            if threshold is None:
                # Default threshold was not overridden by optional keyword argument
                threshold = self.probability_threshold
        else:
            if threshold is None:
                # Recall/precision threshold types must be accompanied by a numerical value for the threshold
                raise ValueError('If threshold_type is ' + threshold_type + ', threshold must be specified.')
            # Get the probability threshold from the recall/precision threshold
            threshold = self.thresholds[(np.abs(np.array(self.metrics[threshold_type]) - threshold)).argmin()]

        # noinspection PyPep8Naming
        # Calculate label sequences for the given probability threshold
        PMGetter = PredictedMarkingsGetter(self, threshold)
        return PMGetter.run()  # Returns a list of struct-like objects containing video number, ant number and marking events frame numbers.
    # endregion

    # region Plotting Methods
    def draw_precision_recall_curve(self, title: str = ''):
        """
        Plot precision-recall curve.
        :param title: Optional title string
        """
        plt.figure('Precision vs Recall curve')
        plt.plot(self.metrics['recall'], self.metrics['precision'],
                 marker='o', label='fold number ' + str(self.validation_fold_index + 1))
        plt.title(title)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show(block=False)
    # endregion
# endregion


# region Threshold Chunk Loop Classes
# region ChunkProcess Abstract Base Class
class ChunkProcess(metaclass=ABCMeta):
    """
    Abstract class for custom processing inside a loop over predicted and actual full trajectory binary label sequences.
    """
    # region Constructor
    @abstractmethod
    def __init__(self, evaluation_object):
        """
        Abstract Base Constructor method for ChunkProcess objects. This abstract method is invoked by the
        constructors of concrete subclasses of ChunkProcess.
        :param evaluation_object: Evaluation object to process.
        """
        self.eval_obj = evaluation_object

        # Placeholder for a numpy array of binary sequences resulting out of thresholding all predicted probabilities
        # sequences with a given threshold.
        self.predicted_labels = None
    # endregion

    # region Chunk Generator
    def _label_chunks(self):
        """
        Private generator yielding all windowed segment thresholded binary label sequences associated with a single
        full trajectory, in order, referred to as a chunk.
        :return: labels_out - Numpy array containing all windowed segment thresholded binary label sequences
        associated with a single full trajectory.
        """
        # noinspection SpellCheckingInspection
        # Find all unique full trajectory indices associated with sequence segments contained in the validation dataset.
        unique_vals = np.unique(self.eval_obj.validation_association)

        for val in unique_vals:
            # For a given full trajectory, find all associated sequence segments indices
            indices_of_trajectory = np.where(self.eval_obj.validation_association == val)
            # Get thresholded binary label sequence segments for the associated sequence segments
            labels_out = self.predicted_labels[indices_of_trajectory]

            # Sort the associated binary label sequence segments by their order in the original full trajectory
            window_indices_in_fold = self.eval_obj.validation_window_indices_in_folds[0]
            labels_out = labels_out[np.argsort(window_indices_in_fold[indices_of_trajectory])]
            yield labels_out
    # endregion

    # region Private Methods
    def _get_full_sequences(self, chunk, actual_full_sequence):
        """
        Private method which generates a predicted binary label sequence of a single full trajectory via
        _final_sequence_from_labeled_windows, compares its length with the actual binary label sequence of the same
        trajectory, and truncates either to make sure they are of the same size
        :param chunk: Numpy array containing multiple overlapping-window binary label sequence segments associated
        with a single full trajectory, in order.
        :param actual_full_sequence: Binary label sequence of the very same full trajectory.
        :return:  predicted_full_sequence - predicted binary label sequence of the full trajectory
                  actual_full_sequence - actual binary label sequence of the full trajectory.
        """
        predicted_full_sequence = self._final_sequence_from_labeled_windows(chunk).astype(float)
        if actual_full_sequence.shape[0] > predicted_full_sequence.shape[0]:
            actual_full_sequence = actual_full_sequence[:predicted_full_sequence.shape[0]]
        elif actual_full_sequence.shape[0] < predicted_full_sequence.shape[0]:
            predicted_full_sequence = predicted_full_sequence[:actual_full_sequence.shape[0]]
        return predicted_full_sequence, actual_full_sequence

    def _final_sequence_from_labeled_windows(self, labels):
        """
        Private method for generation of a single full trajectory binary label sequence from multiple
        overlapping-window binary label sequence segments, by way of majority vote.
        :param labels: Numpy array containing all overlapping-window binary label sequence segments associated with a
                       single full trajectory, in order.
        :return: Numpy array of a single full trajectory binary label sequence.
        """
        labels = labels.astype(int)  # Cast to int for processing
        labels[labels == 0] = -1  # No markings need to carry a negative weight when summing the votes

        # Calculate original trajectory length
        num_windows = labels.shape[0]
        original_sequence_size = self.eval_obj.stride * (num_windows - 1) + self.eval_obj.sample_window_size

        padded_sequence_list = []
        for starting_index, sequence in zip(range(0, num_windows * self.eval_obj.stride, self.eval_obj.stride),
                                            labels):
            # Loop over the starting index of each segment window in the original trajectory, and the binary label
            # segment itself.

            # Embed the binary label sequence segment in a zeros sequence of the full trajectory length,
            # at its appropriate location within the full sequence. This is actually done through padding before and
            # after.
            padded_sequence = np.pad(sequence, ((starting_index, original_sequence_size - (self.eval_obj.sample_window_size + starting_index)), (0, 0)))

            # Append the full-trajectory-long zero-padded sequence to a list
            padded_sequence_list.append(padded_sequence)
        padded_sequence_stack = np.stack(padded_sequence_list, axis=0)  # Stack the list of padded sequences into a numpy array

        # Sum all padded sequences element-by-element to perform a majority voting procedure. Note that each segment
        # does not contribute to the result of the calculation outside its window, because of the zero-padding.
        final = np.add.reduce(padded_sequence_stack, axis=0)

        # Set all positive majority vote (and equal votes) time-points to 1, all negative majority vote to 0 to
        # convert to a binary label sequence, and return it.
        final[final >= 0] = 1
        final[final < 0] = 0
        return final

    def _chunk_loop(self):
        """
        Perform a loop which calculates predicted and actual full trajectory binary label sequences and allow
        further custom subclass within-loop processing.
        """
        lc_gen = self._label_chunks()  # Chunk generator
        for chunk, actual_full_sequence, name, time \
                in zip(lc_gen, self.eval_obj.validation_labels,
                       self.eval_obj.validation_names, self.eval_obj.validation_time):
            # Loop over chunks of predicted single-trajectory-associated label sequence segments, as well as actual
            # label sequences of the respective trajectories, the trajectory names (video and ant numbers),
            # and lists of their frame numbers.

            # Get appropriately-sized full-trajectory-length predicted and actual thresholded binary label sequences
            actual_full_sequence = actual_full_sequence.astype(float).reshape(-1, 1)
            predicted_full_sequence, actual_full_sequence = self._get_full_sequences(chunk, actual_full_sequence)

            # Further custom processing implemented in subclasses
            args = {'predicted_full_sequence': predicted_full_sequence,
                    'actual_full_sequence': actual_full_sequence,
                    'name': name,
                    'time': time}
            self._chunk_loop_internal(args)
    # endregion

    # region Abstract Methods
    @abstractmethod
    def _chunk_loop_internal(self, args):
        """
        Abstract method for further in-loop processing, given full-trajectory-length predicted and actual thresholded
        binary label sequences of a single trajectory in each iteration of the loop.
        :param args: Dictionary containing arguments used by the different concrete chunk loop internal functions
        """
        pass
    # endregion
# endregion


# region PredictedMarkingsGetter Concrete Subclass (ChunkProcess)
class PredictedMarkingsGetter(ChunkProcess):
    """
    Concrete class subclassing ChunkProcess for obtaining all predicted marking events information.
    """
    # region Constructor
    def __init__(self, evaluation_object, threshold):
        """
        Constructor method for the PredictedMarkingsGetter class.
        :param evaluation_object: Associated parent Evaluation object
        :param threshold: Probability threshold to use when thresholding the validation predicted probabilities 
        sequences to get binary label sequences.  
        """
        super().__init__(evaluation_object)  

        self.predicted_labels = self.eval_obj.predicted_probabilities > threshold  # Binarize the validation predicted probabilities sequences
        self.predicted_full_sequence_positives = []  # Initialize the list to be returned 
    # endregion
    
    # region Methods
    def run(self):
        """
        Execute the loop.
        :return: List of PredictedPositives struct-like objects, each associated with a single full trajectory, 
        specifying all marking events frame numbers as well as the video and ant number. 
        """
        super()._chunk_loop()
        return self.predicted_full_sequence_positives
    # endregion
    
    # region Overriding methods
    def _chunk_loop_internal(self, args):
        """
        Get the relevant data for the current predicted full trajectory sequence (marking events, video number, 
        ant number) and store in a list.
        :param args: Dictionary containing input arguments. Here the name (video + ant numbers), time (frame numbers)
         and predicted_full_sequence keys are used.
        """
        # region PredictedPositives Nested Struct-like Class
        class PredictedPositives:
            """
            Nested struct-like class implementing a struct-like behavior, storing information on the marking
            events predicted by the model for a single trajectory.
            """
            # region Constructor
            def __init__(self, name, frames, sequence):
                """
                Constructor Method for PredictedPositives struct-like class.
                :param name: String containing video and ant numbers, formatted video_ant
                :param frames: List of all absolute frame numbers for the trajectory
                :param sequence: Predicted binary label full trajectory sequence
                """
                # Get video and ant numbers from the formatted name string
                self.video = name.split('_')[0]
                self.ant = name.split('_')[1]

                # Get a list of frames in which the model predicts marking events have occurred
                self.positive_frames = frames[np.where(sequence == 1)[0]]
            # endregion
        # endregion

        # Create a PredictedPositives structure for the current trajectory from the input argument dictionary
        predicted = PredictedPositives(args['name'], args['time'], args['predicted_full_sequence'])
        self.predicted_full_sequence_positives.append(predicted)
    # endregion
# endregion


# region ThresholdChunkProcess Abstract Subclass (ChunkProcess)
class ThresholdChunkProcess(ChunkProcess):
    """
    Abstract subclass of ChunkProcess adding an outer loop running over different thresholds to processing the
    probability chunks, with custom processing inside the loop implemented in its concrete subclasses.
    """
    # region Constructor
    @abstractmethod
    def __init__(self, evaluation_object):
        """
        Abstract constructor method for ThresholdChunkProcess objects
        :param evaluation_object: Evaluation object to process.
        """
        super().__init__(evaluation_object)
        # Numpy array defining a list of probability thresholds to use as the outer loop variable in evaluating the
        # performance of the model in terms of precision, recall and F1.
        self.thresholds = evaluation_object.thresholds
    # endregion

    # region Private Methods
    def _threshold_loop(self):
        """
        Perform an outer loop over the input thresholds array, inside which the threshold is used to calculate the
        predicted binary label segment arrays before performing the internal loop over "chunks" of sequence segments,
        each chunk associated with a particular full trajectory. Allow custom in-loop processing before and after the
        internal loop.
        """
        for threshold in self.thresholds:
            self._threshold_loop_internal_initial()  # Perform custom processing, implemented in concrete subclasses
            self.predicted_labels = self.eval_obj.predicted_probabilities > threshold  # Calculate binary label segment arrays
            self._chunk_loop()  # Internal loop over chunks of predicted binary label segment arrays, each chunk associated with a particular full trajectory.
            self._threshold_loop_internal_final()  # Perform custom processing, implemented in concrete subclasses
    # endregion

    # region Abstract Methods
    @abstractmethod
    def _threshold_loop_internal_initial(self):
        """
        Abstract method for custom processing at the beginning of the outer loop iterating over the threshold values.
        """
        pass

    @abstractmethod
    def _threshold_loop_internal_final(self):
        """
        Abstract method for custom processing at the end of the outer loop iterating over the threshold values.
        """
        pass
    # endregion
# endregion


# region MultiThresholdMetricsCalculator Subclass (ThresholdChunkProcess)
class MultiThresholdMetricsCalculator(ThresholdChunkProcess):
    """
    Concrete subclass inheriting from ThresholdChunkProcess for evaluation of model performance in terms of
    precision, recall and F1, at different probability thresholds.
    """
    # region Constructor
    def __init__(self, evaluation_object: Evaluation):
        """
        Constructor method for MultiThresholdMetricsCalculator objects
        :param evaluation_object: Evaluation object to process.
        """
        super().__init__(evaluation_object)
        self.basic_metrics = {'TP': 0, 'FP': 0, 'FN': 0}  # Initialize dictionary storing true positive,
        # false positive, and false negative basic metrics total counts (per threshold)
        self.metrics = {'precision': [], 'recall': [], 'f1': []}  # Initialize dictionary storing lists containing the
        # results of the precision, recall and F1 metrics computation at different probability thresholds.
    # endregion

    # region Methods
    def run(self):
        """
            Execute the loop.
            :return: Dictionary storing lists containing the results of the precision, recall and F1 metrics
            computation at different probability thresholds.
            """
        super()._threshold_loop()
        return self.metrics
    # endregion

    # region Overriding Methods
    def _threshold_loop_internal_initial(self):
        """
        Reset the basic metrics dictionary for the current threshold iteration
        """
        self.basic_metrics = {'TP': 0, 'FP': 0, 'FN': 0}

    def _threshold_loop_internal_final(self):
        """
        Evaluate recall, precision and F1 metrics from the basic true positive, true negative and false negative
        metrics, for the current threshold.
        """
        precision_out, recall_out, f1_out = Evaluation.simple_metrics(self.basic_metrics['TP'],
                                                                      self.basic_metrics['FP'],
                                                                      self.basic_metrics['FN'])
        self.metrics['precision'].append(precision_out)
        self.metrics['recall'].append(recall_out)
        self.metrics['f1'].append(f1_out)

    def _chunk_loop_internal(self, args):
        """
        Calculate and accumulate basic metrics for the current predicted full trajectory binary label sequence
        :param args: Dictionary containing input arguments. Here the predicted_full_sequence and actual_full_sequence
        keys are used.
        """
        # Calculate basic metrics from current predicted full trajectory binary label sequence
        TP_seq, FP_seq, FN_seq = \
            Evaluation.evaluate_sequence(args['predicted_full_sequence'], args['actual_full_sequence'],
                                         self.eval_obj.prediction_window_size)

        # Accumulate results in dictionary (for current threshold)
        self.basic_metrics['TP'] += TP_seq
        self.basic_metrics['FP'] += FP_seq
        self.basic_metrics['FN'] += FN_seq
    # endregion
# endregion
# endregion


# region Other Functions
def print_cross_validation_results(evaluations_list: list[Evaluation], threshold: Optional[float] = None):
    """
    Print into command prompt the metrics results for the different folds at a given probability threshold,
    in a tabulated form.
    :param evaluations_list: List of Evaluation objects associated with the different folds' models.
    :param threshold: Threshold used for binarization of the predicted probability sequences.
    """
    if threshold is None:
        # Default threshold was not overridden by optional keyword argument
        threshold = evaluations_list[0].probability_threshold

    # Create matrix for printing in tabulated form
    thresh_ind = (np.abs(evaluations_list[0].thresholds - threshold)).argmin()
    results_matrix = [[ev[0] + 1,
                       ev[1].metrics['precision'][thresh_ind],
                       ev[1].metrics['recall'][thresh_ind],
                       ev[1].metrics['f1'][thresh_ind]]
                      for ev in enumerate(evaluations_list)]  # fold number, precision, recall, F1

    # Print results to command prompt
    print('\nCross-validation results at probability threshold = ' + str(threshold) + '\n')
    print(tabulate.tabulate(results_matrix, headers=["Fold", "Precision", "Recall", "F1"], tablefmt='grid'))
# endregion


# region DilatedOnTrainingEvaluation Keras Callback Class
class DilatedOnTrainingEvaluation(K_callbacks.Callback):
    """
    Keras Callback Class for custom evaluation of model performance on validation data on end of each training epoch.
    Basic evaluation metrics (true positive, false positive and false negative) are calculated in a tolerant,
    window-based fashion (See evaluation.Evaluation.evaluate_sequence [region Static Methods] for detailed
    explanation.
    """
    # region Constructor
    def __init__(self, model, validation: Data, tolerance: int, probability_threshold: float):
        """
        Constructor method for DilatedOnTrainingEvaluation Objects.
        :param model: Keras model in training.
        :param validation: Validation data to evaluate model with
        :param tolerance: Window size allowing some tolerance built into the definition of what a correct prediction is.
        :param probability_threshold: Threshold used for binarization of the predicted probability sequences.
        """
        super().__init__()
        self.model = model

        # Validation features and labels
        self.val_inputs = validation.windowed_features
        self.val_targets = validation.windowed_labels

        # Evaluation-related Attributes
        self.tolerance = tolerance  # Window size used for tolerant evaluation of basic metrics
        self.threshold = probability_threshold  # Threshold used for binarization of predicted probability sequences
    # endregion

    # region Overriding Methods
    def on_epoch_end(self, epoch, logs=None):
        """
        Evaluates model performance with a custom window-size based tolerant definition of the basic metrics. The
        method is executed after each training epoch by Tensorflow's internal machinery.
        :param epoch:  current epoch (Tensorflow internal variable)
        :param logs: dict containing the loss value and all the metrics at the end of a epoch (Tensorflow internal variable).
        """
        all_predicted = np.array(self.model(self.val_inputs))  # Get model prediction on validation data features

        # Loop over predicted probability and actual sequence segments
        TP = FP = FN = 0
        for predicted_probabilities, target in zip(all_predicted, self.val_targets):
            # Binarize predicted probability sequences using given threshold
            predicted = (predicted_probabilities > self.threshold).astype(float)

            # Compute and accumulate basic metrics over them (true positive, false positive and true negative counts)
            TP_seq, FP_seq, FN_seq = Evaluation.evaluate_sequence(predicted, target, self.tolerance)
            TP += TP_seq
            FP += FP_seq
            FN += FN_seq

        # Calculate precision, recall and f1 metrics for the validation data at the given threshold
        precision, recall, f1 = Evaluation.simple_metrics(TP, FP, FN)

        # Print results to command prompt
        print('val_special_precision:', precision)
        print('val_special_recall:', recall)
        print("val_special_f1:", f1)
    # endregion
# endregion
