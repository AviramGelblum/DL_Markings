from Runner import Runner
from Data import AntData, DataType
from model import Model
from evaluation import Evaluation, print_cross_validation_results
from sklearn.preprocessing import StandardScaler

'''
Script running a Keras-based deep learning module for automatic recognition of ant markings based on trajectory 
features.
'''


def main():
    """
    The script is based on a Runner object which receives a Pipeline object specifying methods and their
    respective arguments, and processes them sequentially.
    """
    data_dir = r'\\phys-guru-cs\ants\Aviram\Marking behavior data\new data'  # Root data directory
    run_object = Runner(data_dir, training_parameters={'n_epoch': 1})
    run_object.load_data()

    # Pipe objects receive a command list of 4-tuples of the format (method, list of method arguments, dictionary of
    # keyword arguments, list of enum types selecting object instances to perform method on(None if irrelevant))
    initial_processing_pipe = [(AntData.get_time_points, [[1]], {}, [DataType.Full]),
                               (AntData.remove_features, [[0, 1]], {}, [DataType.Full]),
                               (AntData.transform_features, [[2, 4]], {'transformer': StandardScaler}, [DataType.Full]),
                               (AntData.get_only_positive_examples, [], {}, [DataType.Full]),
                               (AntData.assure_binary_labels, [], {}, [DataType.Full])]
    run_object.process(initial_processing_pipe)

    cutting_splitting_pipe = [(AntData.cut_sequence_into_windows, [], {}, [DataType.Full]),
                              (AntData.split_test_data, [], {}, [DataType.Full]),
                              (AntData.stratify_cross_validation_folds, [], {}, [DataType.Full])]
    run_object.process(cutting_splitting_pipe)

    after_splitting_pipe = [(AntData.detrend, [[0, 1]], {}, [DataType.Full]),
                            (AntData.add_aggregate_velocity_features, [[2, 4]], {}, [DataType.Full])]

    run_object.process(after_splitting_pipe)

    # [(AntData.dilate_labels, [], {}, [DataType.Training]),
    cross_validation_pipe = [(AntData.expand_label_dims, [], {}, [DataType.Training, DataType.Validation]),
                             (AntData.augment_random_rotations, [[5, 9], [0, 1]], {'augment_factor': 2},
                              [DataType.Training]),
                             (AntData.augment_xy_flip, [[5, 9], [0, 1]], {}, [DataType.Training]),
                             (AntData.separate_circular_features, [[5, 9]], {}, [DataType.Training, DataType.Validation]),
                             (Model.create, [], {}, None),
                             (Model.compile, [], {}, None),
                             (Model.train, [], {}, None),
                             (Model.predict, [], {}, None),
                             (Evaluation.multi_thresh_metrics, [], {}, None),
                             (Evaluation.draw_precision_recall_curve, [], {}, None)]

    run_object.cross_validation_process(cross_validation_pipe)

    print_cross_validation_results(run_object.cv_evaluations)
    # noinspection PyUnusedLocal
    predicted_positives = run_object.evaluation.get_predicted_markings(threshold=0.5, threshold_type='recall')
    # noinspection PyUnusedLocal
    a = 1


if __name__ == "__main__":
    main()
