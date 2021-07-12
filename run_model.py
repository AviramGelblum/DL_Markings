import Loader
import preprocessing as pp
import evaluation as ev
import math


def main():
    DATA_DIR = r'\\phys-guru-cs\ants\Aviram\Marking behavior data'
    if __name__ == '__main__':
        # region Loading and Setup
        get_set_seed = False
        if get_set_seed:
            pp.get_set_randomality()


        loader = Loader.csvLoader(DATA_DIR)
        print('Loading data')
        ant_data = loader.load_data()  # {'header': None}
        ant_data = loader.to_numpy(ant_data)
        features = [mat[:, 2:-1] for mat in ant_data]  # remove name, frame and labels
        labels = [mat[:, -1] for mat in ant_data]  # 3 classes in the data
        # endregion

        # region Preprocessing
        # region Reshaping and Stratifying the Data
        print('Separating sequences with positive examples')
        pos_labels, pos_features, positive_traj_indices, _, _ = \
            pp.separate_any_positive_examples(features, labels)
        original_trajectory_sizes = [f.shape[0] for f in pos_features]

        print('Splitting array into overlapping set-size array, using given window/stride sizes')
        window_size = 12 * pp.fps  # 12 seconds
        stride = int(2.5 * pp.fps)
        pos_features, pos_labels, trajectory_association = \
            pp.cut_array_into_windows(pos_features, pos_labels, window_size, stride)

        print('Separating out test data')
        percent_test = 10
        test, trajectory_association, pos_features, pos_labels = \
            pp.get_test_data(pos_features, pos_labels, trajectory_association, percent_test, original_trajectory_sizes)

        print('Stratifying positive/negative frequency across folds')
        number_of_folds = 5  # trajectory_association.shape[0] - 1  # leave-one-out for now
        training, folds_indices = pp.stratify_training_data(pos_features, pos_labels, trajectory_association,
                                                            number_of_folds)

        validation = pp.create_validation_set(training, folds_indices)  # single
        # fold validation, no cross-validation for now
        # training,  validation, test -  dictionaries with features, labels, trajectory_association
        # endregion

        # region Data Augmentation, Feature and Label Engineering
        print('Smearing positive labels')
        window_size = math.ceil(pp.MARKING_TIME * pp.fps)
        pp.dilate_labels(training, window_size)

        # size of smear window dependent on typical duration of a mark as well as fps of the video
        spatial_indices = [0, 1]
        circular_indices = [3, 5]

        print('Augmenting data (random rotations)')
        random_rotation_thetas = pp.augment_random_rotations(training, circular_indices, spatial_indices,
                                                             augment_factor=2)
        print('Augmenting data (xy flipping)')
        pp.augment_xy_flip(training, circular_indices, spatial_indices)

        print('De-trending y (linear), setting initial x to zero')
        for data in [training, validation, test]:
            pp.detrend(data, spatial_indices)

        print('Recalculating circular features into cos/sin')
        for data in [training, validation, test]:
            pp.separate_circular_features(data, circular_indices)

        velocity_magnitude_indices = [2, 5]
        print('Adding aggregate velocity features (mean, std)')
        for data in [training, validation, test]:
            pp.add_aggregate_velocity_features(data, velocity_magnitude_indices)
        # endregion
        # endregion

        # region Model Training
        # TODO: get model from model file, run
        # endregion

        # region Evaluation

        # TODO: precision-recall curve as a function of positive probability cutoff, AUC

        # output - labeled windows, send all windows related to each sequence (separately),
        # for each original sequence: renumber to -1,1, pad with zeros, add.reduce and return to 0,1,
        # then evaluate (final_sequence_from_labeled_windows)
        TP = FP = FN = 0
        for chunk, actual_full_sequence in zip(ev.label_chunks(predicted_labels, validation['association']),
                                               validation['labels']):
            predicted_full_sequence = ev.final_sequence_from_labeled_windows(chunk, stride)
            TP_seq, FP_seq, FN_seq = ev.evaluate_sequence(predicted_full_sequence, actual_full_sequence)
            TP += TP_seq
            FP += FP_seq
            FN += FN_seq
        precision, recall, F1 = ev.get_metrics(TP, FP, FN)
        # endregion

main()
