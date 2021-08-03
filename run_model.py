import tensorflow.keras.backend as Kbackend
import tensorflow.keras.callbacks as Kcallbacks
import tensorflow.keras.optimizers as Kopt
import tensorflow.keras.metrics as Kmetrics
import Loader
import preprocessing as pp
import evaluation as ev
import math
import model
import tensorflow_addons as tfa
from data_generator import DataGenerator
import numpy as np
from sklearn.preprocessing import StandardScaler



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

        indices_to_transform = [2, 4]
        pp.transform(features, indices_to_transform, StandardScaler)

        # endregion

        # region Preprocessing
        # region Reshaping and Stratifying the Data
        print('Separating sequences with positive examples')
        pos_labels, pos_features, positive_traj_indices, _, _ = \
            pp.separate_any_positive_examples(features, labels)
        for p in pos_labels:
            p[p > 1] = 1
        original_trajectory_sizes = [f.shape[0] for f in pos_features]
        original_number_of_positives = [sum(l) for l in pos_labels]

        print('Splitting array into overlapping set-size array, using given window/stride sizes')
        sample_window_size = pp.divisible_window_size(6 * pp.fps, 81)  # closest to 6 seconds but divisible by 81 for
        # unet purposes
        stride = int(1.25 * pp.fps)

        pos_cut_features, pos_cut_labels, trajectory_association = \
            pp.cut_array_into_windows(pos_features, pos_labels, sample_window_size, stride)



        print('Separating out test data')
        percent_test = 10
        test, trajectory_association, pos_cut_features, pos_cut_labels = \
            pp.get_test_data(pos_cut_features, pos_cut_labels, trajectory_association, percent_test, original_trajectory_sizes)



        print('Stratifying positive/negative frequency across folds')
        number_of_folds = 5  # trajectory_association.shape[0] - 1  # leave-one-out for now
        training, window_indices_in_folds = pp.stratify_training_data(pos_cut_features, pos_cut_labels, trajectory_association,
                                                            original_trajectory_sizes, original_number_of_positives,
                                                            number_of_folds)

        validation = pp.create_validation_set(training, window_indices_in_folds)  # single
        # fold validation, no cross-validation for now
        # training,  validation, test -  dictionaries with features, labels, trajectory_association
        # endregion

        # leave only the windows with positive examples
        # pp.take_only_positive_windows(training)

        # region Data Augmentation, Feature and Label Engineering
        print('Smearing positive labels')
        dilation_window_size = math.ceil(pp.MARKING_TIME * pp.fps)
        pp.dilate_labels(training, dilation_window_size)

        # size of smear window dependent on typical duration of a mark as well as fps of the video
        feature_types = {'Spatial': [0, 1], 'Circular': [3, 5], 'Velocity Magnitude': [2, 4]}

        print('De-trending y (linear), setting initial x to zero')
        for data in [training, validation, test]:
            pp.detrend(data, feature_types['Spatial'])

        print('Adding aggregate velocity features (mean, std)')
        for data in [training, validation, test]:
            pp.add_aggregate_velocity_features(data, feature_types['Velocity Magnitude'])
            data['labels'] = np.expand_dims(data['labels'].astype('float32'), axis=2)

        feature_types['Circular'] = [5, 9]
        feature_types['Velocity Magnitude'] = [2, 6]
        feature_types['Velocity Mean'] = [3, 7]
        feature_types['Velocity STD'] = [4, 8]

        # augmentation is done as part of the random data generation (possibly takes more time but less space,
        # there is an option to not do that and we'll see what's better)
        is_data_from_generator = False
        if not is_data_from_generator:
            print('Augmenting data (random rotations)')
            random_rotation_thetas = pp.augment_random_rotations(training, feature_types['Circular'],
                                                                 feature_types['Spatial'],
                                                                 augment_factor=2)
            print('Augmenting data (xy flipping)')
            pp.augment_xy_flip(training, feature_types['Circular'], feature_types['Spatial'])

            print('Recalculating circular features into cos/sin')
            pp.separate_circular_features(training, feature_types['Circular'])

        for data in [validation, test]:
            pp.separate_circular_features(data, feature_types['Circular'])

        for data in [training, validation, test]:
            data['features'] = data['features'].astype(float)


        pp.show_shapes(training, validation, test)


        # endregion
        # endregion

        # region Model Training
        Kbackend.clear_session()
        # training['features'] = np.delete(training['features'], list(range(0, 8)), axis=2)
        # validation['features'] = np.delete(validation['features'], list(range(0, 8)), axis=2)
        num_features = training['features'].shape[2]
        if is_data_from_generator:
            num_features = num_features+2  # separation to cos,sin is done in the generator
        num_training_examples = training['features'].shape[0]
        input_size = (sample_window_size, num_features)
        maxpool = [3, 3, 3, None]
        dropout = [None, None, 0.5, 0.5]
        # maxpool = [3, None]
        # dropout = [None, 0.5]
        unet = model.UNet1D(input_size, depth=4, kernel_size=3, initial_num_filters=32, dropout=dropout,
                      maxpool=maxpool, BatchNorm=True)

        augment_factor = 9
        batch_size = model.training_parameters['batch_size']
        learning_rate = model.training_parameters['learning_rate']
        n_epoch = model.training_parameters['n_epoch']
        probability_threshold = model.training_parameters['probability_threshold']

        unet.model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25),
                           optimizer=Kopt.Adam(learning_rate=learning_rate),
                           metrics=[Kmetrics.Precision(), Kmetrics.Recall()])

        # shapes and types expected by model
        # unet.model.summary()

        # TODO: add regularization when I have enough data

        lr_schedule = Kcallbacks.LearningRateScheduler(model.lrscheduler)

        if is_data_from_generator:
            train_data_generator = DataGenerator(training, feature_types, batch_size=batch_size,
                                                 augment_factor=augment_factor, shuffle=True)
            validation_data_generator = DataGenerator(validation, feature_types, batch_size=batch_size,
                                                      augment_factor=None, shuffle=True)

            # real_example_size = batch_size // augment_factor  # 6 kinds of augmentations per example (2 rotations,
            # # 2 flips per rotation)
            # num_examples_to_fill = batch_size % augment_factor
            # steps_per_epoch = num_training_examples // (real_example_size + num_examples_to_fill)

            unet.model.fit(train_data_generator,
                           epochs=n_epoch,
                           validation_data=validation_data_generator,
                           callbacks=[lr_schedule, ev.DilatedOnTrainingEvaluation(unet.model, validation)])
        else:
            unet.model.fit(x=training['features'],
                           y=training['labels'],
                           batch_size=batch_size,
                           steps_per_epoch=num_training_examples // batch_size,  # floor division
                           epochs=n_epoch,
                           validation_data=(validation['features'], validation['labels']),
                           callbacks=[lr_schedule, ev.DilatedOnTrainingEvaluation(unet.model, validation)],
                           shuffle=True)

        # TODO: class_weights? - should be handled by focal loss, sample_weights? maybe the centers are more important?
        # endregion

        # region Evaluation

        # TODO: precision-recall curve as a function of positive probability cutoff, AUC, maybe as a callback?

        # output - labeled windows, send all windows related to each sequence (separately),
        # for each original sequence: renumber to -1,1, pad with zeros, add.reduce and return to 0,1,
        # then evaluate (final_sequence_from_labeled_windows)

        validation_full_trajectories = [pos_labels[k] for k in np.unique(validation['association'])]
        predicted_probabilities = np.array(unet.model(validation['features']))
        precision, recall, f1 = ev.multi_thresh_metrics(predicted_probabilities, validation,
                                                        validation_full_trajectories, stride, dilation_window_size * 2)
        ev.draw_PR_curve(precision, recall)
        #print('val_special_precision:', precision)
        #print('val_special_recall:', recall)
        #print("val_special_f1:", f1)

        a=1

        #endregion

main()
