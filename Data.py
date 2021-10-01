from __future__ import annotations

import partitioning
from Pipeline import IProcessable

import pandas as pd
import numpy as np
import math
import copy
import itertools
from enum import Enum, auto
import os
from typing import TYPE_CHECKING
from abc import ABCMeta, abstractmethod

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

if TYPE_CHECKING:
    # Always False, allows compile-time type checking of variables of classes whose importing would raise run-time
    # circular import errors. Could simply use if False here, but this way provides a better understanding of what
    # this does and Pycharm doesn't raise a warning regarding unreachable code.
    from Runner import Runner


# region DataType Enum
class DataType(Enum):
    """
    Enum class containing the possible types of data objects
    """
    Full = auto()  # Basic type, used when creating a Data object from all the loaded data

    # Split types, used when creating data objects from an existing Full Data object
    Training = auto()
    Validation = auto()
    Test = auto()
# endregion


# region Data Abstract Base Class
class Data(IProcessable, metaclass=ABCMeta):
    # noinspection SpellCheckingInspection
    """
        Abstract Base Class for storage and manipulation of data implementing the IProcessable interface allowing
        instances to be processed in a Pipeline.
        """

    # region Constructors
    @abstractmethod
    def __init__(self, data: list[np.array], filenames: list[str], runner: Runner):
        """
        Abstract base Constructor method for Data class. Construct a full Data object. i.e. not one based on a
        data-split from another Data object.
        :param data: List of numpy arrays containing the data
        :param filenames: List of filename full paths
        :param runner: Parent Runner object
        """
        # Data-storing Attributes
        self.labels = [mat[:, -1] for mat in data]  # List of full trajectory label numpy arrays
        self.features = [mat[:, :-1] for mat in data]  # List of full trajectory feature numpy arrays
        self.time = None  # Placeholder for time-points/frame numbers, to be set by get_time_points

        self.names = None  # Placeholder for names of trajectories, formatted as video-number_ant-number
        self.parse_filenames_to_names(filenames)  # Parse full filename paths into the aforementioned name format

        self.windowed_labels = None  # Placeholder for numpy array containing labels of all windowed samples cut from the original trajectories.
        self.windowed_features = None  # Placeholder for numpy array containing features of all windowed samples cut from the original trajectories.
        self.association = None  # Placeholder for a numpy array containing association of each windowed sample with the trajectory from which it was cut

        self.test_trajectory_indices = None  # Placeholder for indices of full trajectories selected to comprise the test dataset
        self.test_windowed_indices = None  # Placeholder for indices of windowed samples cut from the full trajectories assigned to the test dataset

        # Initialization of list of lists of indices of windowed samples, associated with each fold.
        # Length of the list should be equal to self.number_of_folds.
        self.window_indices_in_folds = []

        # Random angles used for rotation-based data augmentation
        self.augmentation_random_thetas = []

        # Parameter Attributes
        # See Runner.Runner [region Basic Parameters and Defaults] for explanation of each parameter
        self.stride = runner.data_parameters['stride']
        self.dilation_window_size = runner.data_parameters['dilation_window_size']
        self.sample_window_size = runner.data_parameters['sample_window_size']
        self.percent_test = runner.training_parameters['percent_test']
        self.number_of_folds = runner.training_parameters['number_of_folds']

        # Other Attributes
        self.history = runner.history  # Dictionary containing information about previous processing
        self._type = DataType.Full  # Enum type of data

    def from_existing(self, window_indices_in_folds, type_in: DataType):
        """
        Construct a Data object from the current Data object
        :param window_indices_in_folds: List of lists of indices of windowed samples, associated with each fold used in
        the new data set we generate.
        :param type_in: DataType enum type to assign to the new Data object
        :return: The new partial Data object
        """
        new_data = copy.copy(self)  # No need for deepcopy because numpy recreates new objects when slicing an array from another array

        # Slice windowed labels, windowed features and association of window samples with original trajectories
        # according to the input windowed samples indices.
        all_indices = list(itertools.chain.from_iterable(window_indices_in_folds))
        new_data.windowed_labels = new_data.windowed_labels[all_indices]
        new_data.windowed_features = new_data.windowed_features[all_indices]
        new_data.association = new_data.association[all_indices]
        new_data.window_indices_in_folds = window_indices_in_folds

        # Slice full labels, full features, names and time points based on the full trajectories from which the
        # windowed samples were cut.
        associated_trajectory_indices = np.unique(new_data.association)
        new_data.features = [self.features[k] for k in associated_trajectory_indices]
        new_data.labels = [self.labels[k] for k in associated_trajectory_indices]
        new_data.names = [self.names[k] for k in associated_trajectory_indices]
        if self.time is not None:
            new_data.time = [self.time[k] for k in associated_trajectory_indices]

        # Set enum DataType type
        if type_in is None:
            type_in = DataType.Full
        if not isinstance(type_in, DataType):
            raise TypeError('type_in must be a DataType Enum')
        new_data._type = type_in

        return new_data
    # endregion

    # region Setter Methods
    def set_to_null(self):
        """
        Set certain attributes to None. This is usually done to save memory and should be called after model
        training/prediction.
        """
        if self._type is DataType.Training:
            self.labels = self.features = self.time = self.names = self.windowed_labels = self.windowed_features = \
                self.association = self.augmentation_random_thetas = None
        elif self._type is DataType.Validation:
            self.features = self.windowed_labels = self.windowed_features = None
        elif self._type is DataType.Full:
            raise ValueError('Full Data objects should not be set to null. Check pipeline.')
    # endregion

    # region Overriding Methods
    def verify_type(self, instance_types: list[Enum]):
        """
        Verify that the current object type is in a list of accepted instance-types. This method is called during
        processing of a Pipeline object. Some methods in the pipe should process only objects of a certain type.
        :param instance_types: List of DataType enum types
        :return: bool flag determining if object's type is in the accepted list
        """
        try:
            # Check if instance_types is iterable (should be a list)
            _ = iter(instance_types)
        except TypeError:
            # If not, there are no type constraints and processing is required
            return True

        if self._type in instance_types:
            return True
        return False
    # endregion

    # region Abstract Methods
    @abstractmethod
    def parse_filenames_to_names(self, filenames):
        """
        Parse full filename paths into a video-number_ant-number format
        :param filenames: List of full filename paths
        """
        pass

    @abstractmethod
    def get_time_points(self, *args, **kwargs):
        """
        Set time points of sequences. e.g. by using numbers given in data, assuming a constant spacing between data
        points etc.
        """
        pass
    # endregion

    # region Trajectory Filtering and Cutting Methods
    def get_only_positive_examples(self):
        """
        Filter out trajectories with no marking events
        """
        num_positives = np.array([sum(l_seq) for l_seq in self.labels])  # List of number of marking events in each trajectory
        where_positives = np.nonzero(num_positives)[0]  # Indices of trajectories with marking events

        # Slice the data to only keep the trajectories which had marking events in them
        self.labels = [self.labels[w] for w in where_positives]
        self.features = [self.features[w] for w in where_positives]
        self.names = [self.names[w] for w in where_positives]
        self.time = [self.time[w] for w in where_positives]

        self.history['removed no-positive sequences'] = True  # Update history

    def cut_sequence_into_windows(self):
        """
        Cut data features and labels into overlapping windows, defined by window size = self.sample_window_size and
        stride = self.stride. Zero-pad full trajectory samples smaller than the designated window size.
        """

        # Initialization
        # noinspection SpellCheckingInspection
        label_list, feature_list, traj_index_list = [], [], []
        association_count = 0

        # Loop over trajectory features and label sequence
        for feature_sequence, label_sequence in zip(self.features, self.labels):
            # Calculate index of last windowed sample of the wanted window size. Advancing the window further along
            # the sample results in samples of size smaller than the wanted window size.
            length = feature_sequence.shape[0]
            last_full_window_ind = max(math.floor((length - self.sample_window_size) / self.stride + 1), 1)

            # Loop over initial indices of the windows along the original sequence
            seq_feature_list, seq_label_list = [], []
            for i in range(0, length, self.stride):
                # Append windowed feature and label sequences segment to their respective lists
                seq_feature_list.append(feature_sequence[i: i + self.sample_window_size, :])
                seq_label_list.append(label_sequence[i: i + self.sample_window_size])

            # Remove last segment to make sure all segments are of length = self.sample_window_size
            seq_feature_list = seq_feature_list[:last_full_window_ind]
            seq_label_list = seq_label_list[:last_full_window_ind]

            # Zero-pad full trajectory samples of length < self.sample_window_size
            if last_full_window_ind == 1:
                seq_feature_list[0] = np.pad(seq_feature_list[0],
                                             ((0, self.sample_window_size - seq_feature_list[0].shape[0]), (0, 0)))
                seq_label_list[0] = np.pad(seq_label_list[0], (0, self.sample_window_size - seq_label_list[0].shape[0]))

            # Add windowed feature and label segments generated from the current trajectory to their respective lists
            feature_list.extend(seq_feature_list)
            label_list.extend(seq_label_list)

            # Store association of windowed segments to original trajectory index
            traj_index_list.extend([association_count] * len(seq_feature_list))
            association_count += 1

        # Convert lists to numpy arrays
        self.windowed_labels = np.stack(label_list).astype(float)
        self.windowed_features = np.stack(feature_list).astype(float)
        self.association = np.array(traj_index_list)

        self.history['windowed data'] = True  # Update history
    # endregion

    # region Feature Manipulation Methods
    def remove_features(self, to_remove):
        """
        Remove selected features from data
        :param to_remove: Indices of columns in data to be removed
        """
        num_features = self.features[0].shape[1]
        to_keep = list(set(range(num_features)) - set(to_remove))
        self.features = [mat[:, to_keep] for mat in self.features]
        self.history['features_removed'] = to_remove  # Update history

    def transform_features(self, indices_to_transform, transformer=StandardScaler):
        """
        Transform selected features
        :param indices_to_transform: List of indices of features to be transformed
        :param transformer: Transformer class to be used, or list of transformer classes of length equal to the
        number of features to be transformed
        """
        if isinstance(transformer, list):
            transformers = transformer
            if len(transformer) != len(indices_to_transform):
                raise ValueError("Number of transformers provided must equal the number of features to be transformed.")
        else:
            transformers = [transformer] * len(indices_to_transform)

        # Loop over features to transform
        for index, transformer in zip(indices_to_transform, transformers):

            transformer_obj = transformer()  # Get transformer object

            # Separate out the feature to be transformed
            only_feature_list = [p[:, index] for p in self.features]
            only_feature_array = np.concatenate(only_feature_list)

            transformer_obj.fit(only_feature_array.reshape(-1, 1))  # Calculate transformation parameters

            # Transform selected feature data in-place
            for sequence in self.features:
                sequence[:, index] = np.squeeze(transformer_obj.transform(sequence[:, index].reshape(-1, 1)))

        # Update history
        self.history['features_transformed'] = indices_to_transform
        self.history['transformer'] = [transformer.__name__ for transformer in transformers]

    # noinspection SpellCheckingInspection
    def detrend(self, spatial_coordinate_indices):
        """
        De-trend spatial coordinates in the data using linear regression.
        :param spatial_coordinate_indices: Indices of features containing spatial coordinates. The program assumes
        only a single pair is given.
        """
        model = LinearRegression()
        sequence_list = []
        for sequence in self.windowed_features:  # Loop over windowed feature segments
            # Get x,y coordinates
            x = sequence[:, spatial_coordinate_indices[0]].reshape(-1, 1)
            y = sequence[:, spatial_coordinate_indices[1]]

            # Fit model and get predicted y-values for our x coordinates
            model.fit(x, y)
            trend = model.predict(x)

            # De-trend the line to be centered around y = 0
            detrended_y = y - trend

            # Shift x values such that initial x is at 0
            x_zero = x - x[0]
            x_zero = np.squeeze(x_zero, axis=1)

            new_spatial_features = np.stack([x_zero, detrended_y], axis=1)  # Repack new calculated features
            sequence = np.insert(sequence, spatial_coordinate_indices, new_spatial_features, axis=1)  # Insert into the original sequence

            # Delete original spatial coordinates features
            columns_to_remove = [c[1] + 1 + c[0] for c in enumerate(spatial_coordinate_indices)]
            sequence_list.append(np.delete(sequence, columns_to_remove, axis=1))
        self.windowed_features = np.stack(sequence_list)  # Stack back into a large numpy array
        self.history['de-trended spatial coordinates'] = spatial_coordinate_indices  # Update history

    def add_aggregate_velocity_features(self, velocity_magnitude_indices, window_size=None):
        """
        Calculate rolling aggregate (mean,std) values over velocity magnitude sequences and add these new features to
        the data.
        :param velocity_magnitude_indices: Indices of features containing velocity magnitude values.
        :param window_size: Integer setting the size of the moving window. Ideally should be odd (checked
        and corrected by the program if not)
        """
        if window_size is None:
            # Default window size was not overridden by optional keyword argument
            window_size = self.dilation_window_size

        # To make sure the windows are symmetric around a point, the window size needs to be odd
        if not window_size % 2:
            window_size += 1

        sequence_list = []
        for sequence in self.windowed_features:  # Loop over windowed feature segments
            # Calculate rolling mean and std values
            sequence_df = pd.DataFrame(sequence[:, velocity_magnitude_indices])
            rolling_mean = sequence_df.rolling(window=window_size, center=True, min_periods=1).mean()
            rolling_std = sequence_df.rolling(window=window_size, center=True, min_periods=1).std()

            # Insert into the original sequence
            sequence = np.insert(sequence, [v + 1 for v in velocity_magnitude_indices], rolling_mean, axis=1)
            std_insert_indices = [k[1] + 2 + k[0] for k in enumerate(velocity_magnitude_indices)]
            sequence_list.append(np.insert(sequence, std_insert_indices, rolling_std, axis=1))
        self.windowed_features = np.stack(sequence_list)  # Stack back into a large numpy array
        self.history['aggregated velocity features added'] = True  # Update history

    def separate_circular_features(self, circular_indices):
        """
        Transform angular variable features into cos, sin dual feature representation.
        :param circular_indices: Indices of features containing angular variables. assume variables are given in degrees
        """
        sequence_list = []
        for sequence in self.windowed_features:  # Loop over windowed feature segments
            rads = [np.radians(sequence[:, index]) for index in circular_indices]  # Convert to radians
            cos_sins = np.concatenate([np.stack([np.cos(r), np.sin(r)], axis=1) for r in rads], axis=1)  # Calculate cos, sin values

            # Insert into the original sequence
            doubled_indices = list(itertools.chain.from_iterable([[c, c] for c in circular_indices]))
            sequence = np.insert(sequence, doubled_indices, cos_sins, axis=1)

            # Delete original angular variable features
            columns_to_remove = [c[1] + 2 + c[0] * 2 for c in enumerate(circular_indices)]
            sequence_list.append(np.delete(sequence, columns_to_remove, axis=1))
        self.windowed_features = np.stack(sequence_list)  # Stack back into a large numpy array
        self.history[r'separated circular features into cos/sin'] = True  # Update history
    # endregion

    # region Label Manipulation Methods
    def assure_binary_labels(self):
        """
        Transform ordinal multi-label sequences into binary label sequences, such that all marking types are
        grouped into a single positive category.
        """
        # Assume that the different positive labels are ordinal and increasing, starting from 1. Categorical data
        # is not handled in this function.
        for sequence in self.labels:
            sequence[np.greater(sequence, 1)] = 1  # If label > 1, set it to 1
        self.history['assured binary labels'] = True  # Update history

    def dilate_labels(self, window_size=None):
        """
        Dilate the positive binary labels. This is done to account for the non-precise temporal nature of the
        marking events and the difficulty in pinpointing the exact moment of the marking event if it does exist.
        :window_size: Integer setting the size of the dilating structuring element. Ideally should be odd (checked
        and corrected by the program if not).
        """
        if window_size is None:
            # Default window size was not overridden by optional keyword argument
            window_size = self.dilation_window_size

        # Loop over windowed label segments and dilate each
        for sequence in self.windowed_labels:
            Data.dilate_sequence(sequence, window_size)
        self.history['dilated labels'] = True  # Update history

    # noinspection SpellCheckingInspection
    @staticmethod
    def dilate_sequence(sequence, window_size):
        """
        Dilate positive values in a binary sequence
        :param sequence: Numpy array sequence
        :param window_size: Size of window used for dilation
        """
        # To make sure the windows are symmetric around a point, the window size needs to be odd
        if not window_size % 2:
            window_size += 1

        # noinspection SpellCheckingInspection
        label_inds = np.nonzero(sequence)[0]  # Find indices of positive values corresponding to marking events
        if label_inds.size:  # If there are marking events
            # Calculate indices of all points that fall within (window_size-1)/2 of a positive point
            i_minus = label_inds - (window_size - 1) / 2
            i_plus = label_inds + (window_size - 1) / 2
            inds_in_windows = [np.arange(max(im, 0), min(ip + 1, sequence.shape[0]), dtype=int) for
                               im, ip in zip(i_minus, i_plus)]  # Account for beginning and end points of sequence
            inds_in_windows = np.unique(np.concatenate(inds_in_windows))  # Account for overlapping

            # Set all values at the calculated indices to 1
            sequence[inds_in_windows] = 1

    def expand_label_dims(self):
        """
        Expand dimension of labels from (x,) to (x,1). Needed for processing further downstream.
        """
        self.windowed_labels = np.expand_dims(self.windowed_labels, axis=2)
        self.history['expanded label dimension'] = True  # Update history
    # endregion

    # region Data Splitting Methods
    def split_test_data(self, percent_test=None):
        """
        Split a portion of the data to set aside as the test dataset, according to a given percentage.  To avoid
        information leakage, we set aside entire trajectories, so that segments of the same trajectory cannot appear
        in both the test and training/validation datasets. Thus, the percentage split is based on the length of the
        trajectories.
        :param percent_test: Percentage of the total length of the dataset to allocate to testing
        """

        # Optional overriding of default percentage through keyword argument
        if percent_test is not None:
            self.percent_test = percent_test

        original_trajectory_sizes = [f.shape[0] for f in self.features]
        # Goal of total length we strive to match
        total_test_trajectory_length = round(self.percent_test * sum(original_trajectory_sizes) / 100)
        # Maximum number of trajectories to include
        max_length = round(self.percent_test * len(original_trajectory_sizes) / 100)

        # Algorithm trying to find the best subset of trajectories satisfying the total trajectory length goal and
        # the maximum number of trajectories, given the set of trajectory lengths (subset sum heuristic solution)
        _, self.test_trajectory_indices = \
            partitioning.find_approximate_subset(original_trajectory_sizes, list(range(len(original_trajectory_sizes))),
                                                 total_test_trajectory_length, max_length)

        # Get windowed samples indices associated with the computed test full trajectory indices
        self.test_windowed_indices = np.where(np.isin(self.association, self.test_trajectory_indices))[0]
        self.history['split test data'] = True  # Update history

    def stratify_cross_validation_folds(self, number_of_folds=None):
        """
        Stratify full trajectories into cross-validation folds, taking into account both the length of the
        trajectories (size of data) and the number of marking events within each (keep minority-majority imbalance
        throughout the folds). Calculate over entire trajectories to prevent information leakage between
        training/validation.
        :param number_of_folds: Number of folds the data is to be divided into
        """

        # Optional overriding of default number of folds through keyword argument
        if number_of_folds is not None:
            self.number_of_folds = number_of_folds

        # Remove test trajectories from the data
        post_test_trajectory_sizes = [f[1].shape[0] for f in enumerate(self.features) if f[0] not in
                                      self.test_trajectory_indices]
        post_test_number_of_positives = [sum(lab[1]) for lab in enumerate(self.labels) if lab[0] not in
                                         self.test_trajectory_indices]

        # Combine trajectory lengths with number of marking events to a single measure by scaling both and summing
        # the scaled variables
        scaler = MinMaxScaler()
        stacked = np.stack([post_test_number_of_positives, post_test_trajectory_sizes]).transpose()
        stacked_trans = scaler.fit_transform(stacked)
        combined_measure = stacked_trans.sum(axis=1).tolist()

        # Goal of sum of the combined measure we strive to get in each fold
        sum_per_set = np.sum(combined_measure) / self.number_of_folds

        # Maximum number of trajectories in each fold
        max_length = len(combined_measure) // self.number_of_folds

        # Algorithm implementing a heuristic solution to the balanced multi-way partition problem, which tries to
        # divide a set of numbers (our combined measure values) to N subsets (given by number_of_folds),
        # where each subset's sum should be as close as possible to the goal (sum_per_set) and the number of elements
        # in each subset should be the same (as much as possible).
        # This computation yields a list of length number_of_folds, wherein each element corresponds to a fold and
        # contains trajectory indices associated with the fold.
        folds_indices_of_post_test_trajectories = \
            partitioning.balanced_multi_way_partition(combined_measure, sum_per_set, max_length, self.number_of_folds)

        # Correct trajectory indices to account for the existence of test dataset trajectories get indices of non-test
        # full trajectories
        no_test_indices = [f for f in range(len(self.labels)) if f not in self.test_trajectory_indices]

        # Loop through all indices and correct
        folds_indices_of_post_test_trajectories_updated = []
        for f in folds_indices_of_post_test_trajectories:
            single_fold_indices = []
            for k in f:
                single_fold_indices.append(no_test_indices[k])
            folds_indices_of_post_test_trajectories_updated.append(single_fold_indices)

        # Loop through folds
        for f in folds_indices_of_post_test_trajectories_updated:
            # Get indices of windowed segments associated with full trajectories in each fold (random permute order)
            self.window_indices_in_folds.append(np.random.permutation(np.where(np.isin(self.association, f))[0]))
        self.history['stratified cross-validation folds'] = True  # Update history
    # endregion

    # region Data Augmentation Methods
    def augment_random_rotations(self, circular_indices, spatial_coordinate_indices, augment_factor=2):
        """
        Augment data by way of random rotation of the spatial coordinates
        :param circular_indices: Indices of angular features which should be shifted when rotating the spatial
        coordinates
        :param spatial_coordinate_indices: Indices of features containing spatial coordinates. The program assumes
        only a single pair is given.
        :param augment_factor: Multiplication factor by which the data is augmented. e.g. specifying 2 for this
        keyword argument would create augmented data of size X2 of the original data.
        """

        num_series_examples = self.windowed_features.shape[0]  # Windowed segments data size
        for _ in range(augment_factor):
            new_features = copy.copy(self.windowed_features[:num_series_examples, :, :])
            random_thetas_new = 360 * np.random.random(num_series_examples)  # Generate random rotation angles

            # Rotate each windowed segment data by a random angle
            for sequence, random_theta in zip(new_features, random_thetas_new):
                # Rotate angular feature
                for index in circular_indices:
                    sequence[:, index] = (sequence[:, index] + random_theta) % 360

                # Rotate spatial coordinates
                xy = sequence[:, spatial_coordinate_indices]
                c, s = np.cos(np.radians(random_theta)), np.sin(np.radians(random_theta))
                j = np.array([[c, -s], [s, c]])
                sequence[:, spatial_coordinate_indices] = np.matmul(j, xy.T).T

            self.augmentation_random_thetas.append(random_thetas_new)  # Store random angles used for rotation
            self.windowed_features = np.concatenate([self.windowed_features, new_features])  # Add generated example features
        self.windowed_labels = np.tile(self.windowed_labels, [augment_factor + 1, 1, 1])  # Add labels for generated examples
        self.augmentation_random_thetas = np.concatenate(self.augmentation_random_thetas)  # Concatenate into numpy array
        self.history['random rotation augmentation'] = True  # Update history

    # noinspection SpellCheckingInspection
    def augment_xy_flip(self, circular_indices, spatial_coordinate_indices):
        """
        Augment data by way of flipping the spatial coordinates.
        :param circular_indices: Indices of angular features to shift when flipping
        :param spatial_coordinate_indices: Indices of features containing spatial coordinates to recalculate when
        flipping
        """
        # Two types of flips are performed:
        # x -> -x (flip angle to 180-angle and x spatial coordinate [0] to -x)
        # y -> -y (flip angle to -angle and y spatial coordinate [1] to -y)
        theta_flip = [180, 0]
        spatial_flip_ind = [0, 1]

        num_flips = len(theta_flip)
        num_series_examples = self.windowed_features.shape[0]  # Windowed segments data size
        for thet, spat_ind in zip(theta_flip, spatial_flip_ind):
            new_features = copy.copy(self.windowed_features[:num_series_examples])
            for sequence in new_features:  # Flip each windowed segment
                # Flip angle
                for index in circular_indices:
                    sequence[:, index] = (thet - sequence[:, index]) % 360

                # Flip spatial coordinate
                sequence[:, spatial_coordinate_indices[spat_ind]] = -sequence[:, spatial_coordinate_indices[spat_ind]]
            self.windowed_features = np.concatenate([self.windowed_features, new_features])  # Add generated example features
        self.windowed_labels = np.tile(self.windowed_labels, [num_flips + 1, 1, 1])  # Add labels for generated examples
        self.history['XY flip augmentation'] = True  # Update history
    # endregion
# endregion


# region AntData Concrete Subclass (Data)
class AntData(Data):
    """
    Concrete subclass of the Data base class implementing specific name parsing and frame number calculation
    """

    # region Constructor
    def __init__(self, data: list[np.array], filenames: list[str], runner: Runner):
        """
        Constructor method for the AntData subclass
        :param data: List of numpy arrays containing the data
        :param filenames: List of filename full paths
        :param runner: Parent Runner object
        """
        super().__init__(data, filenames, runner)
    # endregion

    # region Overriding Methods
    def parse_filenames_to_names(self, filenames):
        """
        Parse full filename paths into a video-number_ant-number format
        :param filenames: List of full filename paths
        """
        splits = [os.path.split(filename) for filename in filenames]
        ants = [s[1].split('.')[0][3:] for s in splits]
        videos = [os.path.split(s[0])[1] for s in splits]
        self.names = [v + '_' + a for v, a in zip(videos, ants)]

    def get_time_points(self, index):
        """ Set time points of sequences by using frame numbers given in data
            :param index: Index of column in feature matrix containing frame numbers
        """
        self.time = [mat[:, index] for mat in self.features]
    # endregion
# endregion


# region Helper Functions
def divisible_integer(orig_integer, divider):
    """
    Calculate the next closest integer to an input integer "orig_integer" divisible (without remnant) by another input
    "divider" integer.
    :param orig_integer: Integer for which we find the next closest integer divisible by "divider"
    :param divider: The integer by which the output value must be divisible (without remnant)
    :return: The integer result of the calculation
    """
    return orig_integer + (divider - orig_integer % divider)
# endregion
