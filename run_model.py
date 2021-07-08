import numpy as np
import Loader
import preprocessing
import evaluation

def label_chunks(labels, association_values):
    unique_vals = np.unique(association_values)
    for val in unique_vals:
        yield labels[np.where(association_values == val), :]

def main():
    DATA_DIR = r'\\phys-guru-cs\ants\Aviram\Marking behavior data'
    if __name__ == '__main__':
        get_set_seed = False
        if get_set_seed:
            preprocessing.get_set_randomality()

        # starttime = timeit.default_timer()
        loader = Loader.csvLoader(DATA_DIR)
        print('Loading data')
        ant_data = loader.load_data() # {'header': None}
        ant_data = loader.to_numpy(ant_data)
        features = [mat[:, 2:-1] for mat in ant_data]  # remove name, frame and labels
        labels = [mat[:, -1] for mat in ant_data]  # 3 classes in the data

        print('Separating sequences with positive examples')
        pos_labels, pos_features, positive_traj_indices, _, _ = \
            preprocessing.separate_any_positive_examples(features, labels)
        original_trajectory_sizes = [f.shape[0] for f in pos_features]

        print('Splitting array into overlapping set-size array, using given window/stride sizes')
        window_size = 12*preprocessing.fps  # 12 seconds
        stride = int(2.5*preprocessing.fps)
        pos_features, pos_labels, trajectory_association = \
            preprocessing.cut_array_into_windows(pos_features, pos_labels, window_size, stride)

        print('Separating out test data')
        percent_test = 10
        test, trajectory_association, pos_features, pos_labels = \
            preprocessing.get_test_data(pos_features, pos_labels, trajectory_association, percent_test, original_trajectory_sizes)

        print('Stratifying positive/negative frequency across folds')
        number_of_folds = 5 #trajectory_association.shape[0] - 1  # leave-one-out for now
        # Training and validation folds in training data Folds (?) X N_per_Fold X Timesteps X Features&Labels
        training,  folds_indices = preprocessing.stratify_training_data(pos_features, pos_labels, trajectory_association,
                                                                        number_of_folds)

        validation = preprocessing.create_validation_set(training, folds_indices)  # single
        # fold validation, no cross-validation for now
        # training,  validation, test -  dictionaries with features, labels, trajectory_association

        print('Smearing positive labels')
        preprocessing.smear_labels(training)

        # size of smear window dependent on typical duration of a mark as well as fps of the video
        spatial_indices = [0, 1]
        circular_indices = [3, 5]

        print('Augmenting data (random rotations)')
        random_rotation_thetas = preprocessing.augment_random_rotations(training, circular_indices, spatial_indices,
                                                                        augment_coefficient=2)

        random_mirroring = preprocessing.augment_mirroring(training, circular_indices, spatial_indices,
                                                           augment_coefficient=2)
        print('Recalculating circular features into cos/sin')
        preprocessing.separate_circular_features(training, circular_indices)
        preprocessing.separate_circular_features(validation, circular_indices)
        preprocessing.separate_circular_features(test, circular_indices)





        # TODO: get model from model file, run

        # output - labeled windows, send all windows related to each sequence (separately),
        # for each original sequence: renumber to -1,1, pad with zeros, add.reduce and return to 0,1,
        # then evaluate (final_sequence_from_labeled_windows)
        for chunk, actual_full_sequence in zip(label_chunks(predicted_labels, validation['association']),
                                               validation['labels']):
            predicted_full_sequence = evaluation.final_sequence_from_labeled_windows(chunk, stride)
            evaluation.evaluate_sequence(predicted_full_sequence, actual_full_sequence)




main()
