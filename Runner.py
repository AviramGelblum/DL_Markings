from enum import Enum
import Loader
import Data
import math
import copy
import itertools
import model
from Pipeline import Pipeline
import tensorflow.keras.optimizers as Kopt
import tensorflow.keras.metrics as Kmetrics
import tensorflow_addons as tfa
import numpy as np
import pickle
import evaluation

MARKING_TIME = 0.1  # typical marking time in sec
fps = 50


class RunnerState(Enum):
    Initial = 0
    Loaded = 1
    InitialPreprocessed = 2
    SplitCut = 3
    CrossValidatedRun = 4

    @staticmethod
    def all_states():
        return [v for v in RunnerState.__members__.values()]


class Runner:
    _default_model_parameters = {'dilation_window_size': math.ceil(MARKING_TIME * fps), 'stride': int(1.25 * fps),
                                 'sample_window_size': Data.divisible_window_size(6 * fps, 81)}
    _default_compile_parameters = {'loss': tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25),
                                   'optimizer': Kopt.Adam(learning_rate=0.0005),
                                   'metrics': [Kmetrics.Precision(), Kmetrics.Recall()]}
    _default_training_parameters = {'percent_test': 10, 'do_cross_validation': True, 'number_of_folds': 5,
                                     'n_epoch': 7, 'batch_size': 32, 'probability_threshold': 0.25,
                                     'prediction_window_size': math.ceil(MARKING_TIME * fps) * 3}

    def __init__(self, data_dir, data_format='csv', model_type='unet', model_parameters=None,
                 training_parameters=None, compile_parameters=None):
        self.data_dir = data_dir
        self.data_format = data_format
        self.data = None

        self.model_type = model_type
        self.model = None
        self.evaluation = None
        self.cv_evaluations = []

        self._state = RunnerState.Initial
        self.history = PrettyDict()

        def copy_from_default_if_none(input_object_parameters, default_parameters):
            if input_object_parameters is None:
                object_parameters = default_parameters
            else:
                default_parameters_copy = copy.copy(default_parameters)
                default_parameters_copy.update(input_object_parameters)
                object_parameters = default_parameters_copy
            return object_parameters

        self.training_parameters = copy_from_default_if_none(training_parameters, Runner._default_training_parameters)
        self.model_parameters = copy_from_default_if_none(model_parameters, Runner._default_model_parameters)
        self.compile_parameters = copy_from_default_if_none(compile_parameters, Runner._default_compile_parameters)

    def advance_state(self):
        all_states = RunnerState.all_states()
        current_index = all_states.index(self._state)
        if current_index < len(all_states)-1:
            self._state = all_states[current_index + 1]

    def load_data(self, read_kwargs=None):
        print('Loading data')
        loader = Loader.DataLoader.loader_factory(self.data_format, self.data_dir)
        self.data = loader.load_data(read_kwargs)

        # assume working with numpy, could be changed to accommodate other libraries (pandas etc. if needed)
        self.data = Data.AntData(loader.to_numpy(self.data), loader.filenames, self)
        self._state = RunnerState.Loaded

    def prepare_model(self):
        input_size = (self.model_parameters['sample_window_size'], self.data.windowed_features.shape[2]+6)
        self.model = model.model_factory(self.model_type, input_size, self)
        self.model.compile(self.compile_parameters)

    def prepare_evaluation(self):
        self.evaluation = evaluation.Evaluation(self)
        self.cv_evaluations.append(self.evaluation)

    def process(self, commands_list):
        pipe = Pipeline(commands_list)
        pipe.process([self.data, self.model, self.evaluation])
        self.advance_state()

    def cross_validation_process(self, commands_list):
        pipe = Pipeline(commands_list)
        folds = list(range(self.training_parameters['number_of_folds']))
        for validation_fold_index in folds:
            training_folds = [self.data.window_indices_in_folds[fold] for fold in folds if fold != validation_fold_index]
            training_window_indices = list(itertools.chain.from_iterable(training_folds))
            current_training_data = self.data.from_existing(training_window_indices, Data.DataType.Training)
            validation_window_indices = self.data.window_indices_in_folds[validation_fold_index]
            current_validation_data = self.data.from_existing(validation_window_indices, Data.DataType.Validation)
            self.model.training_data = current_training_data
            self.model.validation_data = current_validation_data
            self.model.validation_fold_index = validation_fold_index
            self.prepare_evaluation()
            pipe.process([current_training_data, current_validation_data, self.model, self.evaluation])
        self._state = RunnerState.CrossValidatedRun

    @staticmethod
    def get_set_randomality():
        filename = 'pickled_random_state.pic'
        try:
            with open(filename, 'rb') as f:
                prng = pickle.load(f)
            state = prng.get_state()
            np.random.set_state(state)
        except FileNotFoundError:
            prng = np.random.RandomState()
            with open(filename, 'wb') as f:
                pickle.dump(prng, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            raise error


class PrettyDict(dict):
    def __repr__(self):
        max_item_index = len(self)-1
        repr_out = '\nProcessing history: \n{'
        for num, kv in enumerate(self.items()):
            if num < max_item_index:
                extra = '\n'
            else:
                extra = '}\n'
            repr_out = repr_out + '{} : {}'.format(kv[0], kv[1]) + extra
        return repr_out