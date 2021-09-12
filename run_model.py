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


