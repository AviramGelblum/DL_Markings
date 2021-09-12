# import copy
#
# import tensorflow.keras.backend as Kbackend
# import tensorflow.keras.callbacks as Kcallbacks
# import tensorflow.keras.optimizers as Kopt
# import tensorflow.keras.metrics as Kmetrics
# import Loader
# import preprocessing as pp
# import evaluation as ev
# import model
# import tensorflow_addons as tfa
# from data_generator import DataGenerator
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
#
#
# def main_old():
#     DATA_DIR = r'\\phys-guru-cs\ants\Aviram\Marking behavior data\new data'
#     if __name__ == '__main__':
#         # region Loading and Setup
#         get_set_seed = False
#         if get_set_seed:
#             pp.get_set_randomality()
#
#
#         loader = Loader.csvLoader(DATA_DIR)
#         print('Loading data')
#         ant_data = loader.load_data()  # {'header': None}
#         ant_data = loader.to_numpy(ant_data)
#         features = [mat[:, 2:-1] for mat in ant_data]  # remove name, frame and labels
#         labels = [mat[:, -1] for mat in ant_data]  # 3 classes in the data
#
#         indices_to_transform = [2, 4]
#         pp.transform(features, indices_to_transform, StandardScaler)
#
#         # endregion
#
#         # region Preprocessing
#         # region Reshaping and Stratifying the Data
#         print('Separating sequences with positive examples')
#         pos_labels, pos_features, positive_traj_indices, _, _ = \
#             pp.separate_any_positive_examples(features, labels)
#         for p in pos_labels:
#             p[p > 1] = 1
#         original_trajectory_sizes = [f.shape[0] for f in pos_features]
#         original_number_of_positives = [sum(l) for l in pos_labels]
#
#         print('Splitting array into overlapping set-size array, using given window/stride sizes')
#         pos_cut_features, pos_cut_labels, trajectory_association = \
#             pp.cut_array_into_windows(pos_features, pos_labels, model.model_parameters['sample_window_size'],
#                                       model.model_parameters['stride'])
#
#
#         print('Separating out test data')
#         percent_test = 10
#         test, trajectory_association, pos_cut_features, pos_cut_labels = \
#             pp.get_test_data(pos_cut_features, pos_cut_labels, trajectory_association, percent_test, original_trajectory_sizes)
#
#
#
#         print('Stratifying positive/negative frequency across folds')
#         number_of_folds = 5  # trajectory_association.shape[0] - 1  # leave-one-out for now
#         training, window_indices_in_folds = pp.stratify_training_data(pos_cut_features, pos_cut_labels,
#                                                                        trajectory_association,
#                                                                     original_trajectory_sizes, original_number_of_positives,
#                                                                     number_of_folds)
#         title = 'no dilation 5 epochs'
#         do_cross_validation = True
#         if do_cross_validation:
#             precision_at_thresh = []
#             recall_at_thresh = []
#             f1_at_thresh = []
#             for fold_num in range(number_of_folds):
#                 training_copy = copy.copy(training)
#                 validation = pp.create_validation_set(training_copy, window_indices_in_folds, fold_num)
#                 feature_types = feature_engineering(training_copy, validation, test)
#                 unet = model_training(training_copy, validation, feature_types)
#
#                 validation_full_trajectories = [pos_labels[k] for k in np.unique(validation['association'])]
#                 predicted_probabilities = np.array(unet.model(validation['features']))
#                 precision, recall, f1, thresholds = ev.multi_thresh_metrics(predicted_probabilities, validation,
#                                                                             validation_full_trajectories,
#                                                                             model.model_parameters['stride'],
#                                                                             window_indices_in_folds[fold_num],
#                                                                             model.model_parameters['prediction_window_size'])
#                 ev.draw_precision_recall_curve(precision, recall, str(fold_num+1))
#
#                 model_thresh_ind = (np.abs(thresholds - model.training_parameters['probability_threshold'])).argmin()
#                 precision_at_thresh.append(precision[model_thresh_ind])
#                 recall_at_thresh.append(recall[model_thresh_ind])
#                 f1_at_thresh.append(f1[model_thresh_ind])
#             plt.legend()
#             ev.print_results(precision_at_thresh, recall_at_thresh, f1_at_thresh)
#         else:
#             n_epochs = [2, 4, 8, 16, 32]
#             for n in n_epochs:
#                 training_copy = copy.copy(training)
#                 validation = pp.create_validation_set(training_copy, window_indices_in_folds, fold_num=0)
#                 feature_types = feature_engineering(training_copy, validation, test)
#                 unet = model_training(training_copy, validation, feature_types, n)
#                 validation_full_trajectories = [pos_labels[k] for k in np.unique(validation['association'])]
#                 predicted_probabilities = np.array(unet.model(validation['features']))
#                 precision, recall, f1, thresholds = ev.multi_thresh_metrics(predicted_probabilities, validation,
#                                                                             validation_full_trajectories,
#                                                                             model.model_parameters['stride'],
#                                                                             window_indices_in_folds[0],
#                                                                             model.model_parameters['prediction_window_size'])
#                 ev.draw_precision_recall_curve(precision, recall, str(n))
#
#         recall_threshold = 0.6
#         recall_index = next(x[0] for x in enumerate(recall.__reversed__()) if x[1] > recall_threshold)
#         probability_threshold = thresholds[recall_index]
#         full_predicted_sequences = ev.get_predicted_markings(predicted_probabilities, validation, validation_full_trajectories,
#                                                              model.model_parameters['stride'], window_indices_in_folds[0],
#                                                              probability_threshold)
#
#
#
#     st = 1
#
# def feature_engineering(training, validation, test):
#     print('Smearing positive labels')
#
#     pp.dilate_labels(training, model.model_parameters['dilation_window_size'])
#
#     # size of smear window dependent on typical duration of a mark as well as fps of the video
#     feature_types = {'Spatial': [0, 1], 'Circular': [3, 5], 'Velocity Magnitude': [2, 4]}
#
#     print('De-trending y (linear), setting initial x to zero')
#     for data in [training, validation, test]:
#         pp.detrend(data, feature_types['Spatial'])
#
#     print('Adding aggregate velocity features (mean, std)')
#     for data in [training, validation, test]:
#         pp.add_aggregate_velocity_features(data, feature_types['Velocity Magnitude'])
#         data['labels'] = np.expand_dims(data['labels'].astype('float32'), axis=2)
#
#     feature_types['Circular'] = [5, 9]
#     feature_types['Velocity Magnitude'] = [2, 6]
#     feature_types['Velocity Mean'] = [3, 7]
#     feature_types['Velocity STD'] = [4, 8]
#
#     # augmentation is done as part of the random data generation (possibly takes more time but less space,
#     # there is an option to not do that and we'll see what's better)
#     if not model.training_parameters['is_data_from_generator']:
#         print('Augmenting data (random rotations)')
#         _ = pp.augment_random_rotations(training, feature_types['Circular'], feature_types['Spatial'], augment_factor=2)
#
#         print('Augmenting data (xy flipping)')
#         pp.augment_xy_flip(training, feature_types['Circular'], feature_types['Spatial'])
#
#         print('Recalculating circular features into cos/sin')
#         pp.separate_circular_features(training, feature_types['Circular'])
#
#     for data in [validation, test]:
#         pp.separate_circular_features(data, feature_types['Circular'])
#
#     for data in [training, validation, test]:
#         data['features'] = data['features'].astype(float)
#
#     # leave only the windows with positive examples
#     # pp.take_only_positive_windows(training)
#
#     pp.show_shapes(training, validation, test)
#
#     # endregion
#     return feature_types
#
#
# def model_training(training, validation, feature_types):
#     Kbackend.clear_session()
#     # training['features'] = np.delete(training['features'], list(range(0, 8)), axis=2)
#     # validation['features'] = np.delete(validation['features'], list(range(0, 8)), axis=2)
#     num_features = training['features'].shape[2]
#     if model.training_parameters['is_data_from_generator']:
#         num_features = num_features + 2  # separation to cos,sin is done in the generator
#     num_training_examples = training['features'].shape[0]
#     input_size = (model.model_parameters['sample_window_size'], num_features)
#     # maxpool = [3, None]
#     # dropout = [None, 0.5]
#     unet = model.UNet1D(input_size)
#
#     batch_size = model.training_parameters['batch_size']
#     unet.model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=model.training_parameters['focal_loss_alpha']),
#                        optimizer=Kopt.Adam(learning_rate=model.training_parameters['learning_rate']),
#                        metrics=[Kmetrics.Precision(), Kmetrics.Recall()])
#
#     # shapes and types expected by model
#     # unet.model.summary()
#
#     lr_schedule = Kcallbacks.LearningRateScheduler(model.lrscheduler)
#
#     if model.training_parameters['is_data_from_generator']:
#         augment_factor = 9
#         train_data_generator = DataGenerator(training, feature_types, batch_size,
#                                              augment_factor=augment_factor, shuffle=True)
#         validation_data_generator = DataGenerator(validation, feature_types, batch_size,
#                                                   augment_factor=None, shuffle=True)
#
#         unet.model.fit(train_data_generator,
#                        epochs=model.training_parameters['n_epoch'],
#                        validation_data=validation_data_generator,
#                        callbacks=[lr_schedule, ev.DilatedOnTrainingEvaluation(unet.model, validation)])
#     else:
#         unet.model.fit(x=training['features'],
#                        y=training['labels'],
#                        batch_size=model.training_parameters['batch_size'],
#                        steps_per_epoch=num_training_examples // batch_size,  # floor division
#                        epochs=model.training_parameters['n_epoch'],    #model.training_parameters['n_epoch'],
#                        validation_data=(validation['features'], validation['labels']),
#                        callbacks=[lr_schedule, ev.DilatedOnTrainingEvaluation(unet.model, validation)],
#                        shuffle=True)
#     return unet


from Runner import Runner
from Data import Data, DataType
from model import Model
from evaluation import Evaluation, print_cross_validation_results
from sklearn.preprocessing import StandardScaler


def main():
    data_dir = r'\\phys-guru-cs\ants\Aviram\Marking behavior data\new data'
    run_object = Runner(data_dir, training_parameters={'n_epoch': 1})
    run_object.load_data()

    initial_processing_pipe = [(Data.remove_features, [[0, 1]], {}, [DataType.Full]),
                               (Data.transform_features, [[2, 4]], {'transformer': StandardScaler}, [DataType.Full]),
                               (Data.get_only_positive_examples, [], {}, [DataType.Full]),
                               (Data.assure_binary_labels, [], {}, [DataType.Full])]
    run_object.process(initial_processing_pipe)

    cutting_splitting_pipe = [(Data.cut_sequence_into_windows, [], {}, [DataType.Full]),
                              (Data.split_test_data, [], {}, [DataType.Full]),
                              (Data.stratify_training_data, [], {}, [DataType.Full])]
    run_object.process(cutting_splitting_pipe)

    run_object.prepare_model()

    cross_validation_pipe = [(Data.dilate_labels, [], {}, [DataType.Training]),
                             (Data.detrend, [[0, 1]], {}, [DataType.Training, DataType.Validation]),
                             (Data.add_aggregate_velocity_features, [[2, 4]], {},
                              [DataType.Training, DataType.Validation]),
                             (Data.expand_label_dims, [], {}, [DataType.Training, DataType.Validation]),
                             (Data.augment_random_rotations, [[5, 9], [0, 1]], {'augment_factor': 2},
                              [DataType.Training]),
                             (Data.augment_xy_flip, [[5, 9], [0, 1]], {}, [DataType.Training]),
                             (Data.separate_circular_features, [[5, 9]], {}, [DataType.Training, DataType.Validation]),
                             (Model.train, [], {}, None),
                             (Model.predict, [], {}, None),
                             (Evaluation.multi_thresh_metrics, [], {}, None),
                             (Evaluation.draw_precision_recall_curve, [], {}, None)]

    run_object.cross_validation_process(cross_validation_pipe)

    print_cross_validation_results(run_object.cv_evaluations)
    predicted = run_object.evaluation.get_predicted_markings(threshold=0.5, threshold_type='recall')
    a=1
if __name__ == "__main__":
    main()


