import Loader
import Data
import model
import evaluation
from Pipeline import Pipeline

from enum import Enum
import math
import numpy as np
import pickle
from typing import Optional, Callable

import tensorflow.keras.optimizers as K_opt
import tensorflow.keras.metrics as K_metrics
import tensorflow_addons as tfa


# region RunnerState Enum
class RunnerState(Enum):
    """Enum class containing the possible states of runner objects."""
    Initial = 0
    Loaded = 1
    InitialPreprocessed = 2
    SplitCut = 3
    CrossValidatedRun = 4

    @staticmethod
    def all_states():
        """
        :return: A list of all state enums
        """
        return [v for v in RunnerState.__members__.values()]
# endregion


# region Runner Class
class Runner:
    """
    Class designed to load data, receive a pipeline of data manipulation/ model/ evaluation methods and process them
    sequentially.
    """

    # region Basic Parameters and Defaults
    MARKING_DURATION = 0.1  # Typical marking time in sec
    fps = 50  # Video frame rate in frames per second
    FRAME_MARKING_DURATION = math.ceil(MARKING_DURATION * fps)  # Typical marking time in frames, used as a basic timescale

    _default_data_parameters = {'dilation_window_size': FRAME_MARKING_DURATION, 'sample_window_size':
                                Data.divisible_integer(6 * fps, 81), 'stride': int(1.25 * fps)}
    '''
    Parameters used during data preprocessing.
    
    dilation_window_size: Size of window used for dilating the (positive) marking events in the label sequences, See 
    Data.Data.dilate_labels method [region Label Manipulation]. The same window size is also used when computing moving 
    mean and standard deviation velocity magnitude features, see Data.Data.add_aggregate_velocity_features method [
    region Feature Manipulation].
    
    sample_window_size: size of moving window used to cut the full trajectory sequences into segments. 
    
    stride: stride of moving window applied when cutting full trajectory sequences into segments.
    '''

    _default_compile_options = {'loss': tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25),
                                'optimizer': K_opt.Adam, 'learning_rate': 0.0005,
                                'metrics': [K_metrics.Precision(), K_metrics.Recall()]}
    '''
    Options used for model compiling.
    See https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile for details about allowed types for each 
    compile option, as well as other options that can be added.
    
    loss: Instantiated Tensorflow/Keras loss function object 
    
    optimizer: Tensorflow/Keras Optimizer class (uninstantiated)
    
    learning_rate: Initial learning rate used for training by the model optimizer 
    
    metrics: Evaluation metrics reported during training (after each epoch).
    '''

    _default_training_parameters = {'percent_test': 10, 'number_of_folds': 5, 'n_epoch': 7, 'batch_size': 32,
                                    'probability_threshold': 0.25, 'discard_data': True,
                                    'prediction_window_size': FRAME_MARKING_DURATION * 3}
    '''
    Parameters used during model training/validation.
    
    percent_test: Percentage of dataset allocated for testing after model training/validation is done.
    
    number_of_folds: Number of folds used during cross-validation. Sets the relative size of the validation 
    dataset, if no cross-validation is used.
    
     n_epoch: Number of epochs to train the model.
     
     batch_size: Size of mini-batch used for model training. 
     
     probability_threshold: Probability threshold used to make a binary choice for each time-point in the 
     predicted probabilities sequence. Used to train and evaluate model performance.
     
     stream_data: Boolean flag determining whether (False) to keep all training and validation data stored in each 
     trained model (during cross-validation), or (True) discard the training and validation data and only keep the 
     attributes of the trained model relevant for further model evaluation.
     
     prediction_window_size: Size of window for dilating the (positive) marking events in the predicted/actual label 
     sequences when deciding if a prediction is true positive, false positive or false negative (for now assessment of 
     True Negative is of little interest). This window is larger than the dilation window size since a predicted 
     marking only needs to be in the time-region of an actual marking, whereas dilation for training is aimed at 
     helping the model to learn the features of the marking behavior itself. 
    '''
    # endregion

    # region Constructor
    def __init__(self, data_dir: str, data_format: str = 'csv', model_type: str = 'unet',
                 data_parameters: Optional[dict] = None,
                 training_parameters: Optional[dict] = None,
                 compile_options: Optional[dict] = None):
        """
        Constructor method for Runner class
        :param data_dir: String containing the full path to the data directory (can have multiple sub-folders)
        :param data_format: String containing the format of the data (file extension)
        :param model_type: String containing the type of the underlying model (see model:model_factory.model_class_dict
         [region Model Factory Function] for allowed types)
        :param data_parameters: Dictionary overriding parameters related to data pre-processing
        :param training_parameters: Dictionary overriding parameters related to model training
        :param compile_options: Dictionary overriding options related to model compilation
        """
        # Data Attributes
        self.data_dir = data_dir  # Full path of the directory containing the data (can contain subdirectories)
        self.data_format = data_format  # Format of the files containing the data.  e.g. csv, txt, etc.
        self.data = None  # Placeholder for a Data object

        # Model Attributes
        self.model_type = model_type  # Type of model to use, e.g. unet
        self.model = None  # Placeholder for a Model object
        self.cv_models = []  # List containing all model objects in a cross-validation run

        # Evaluation Attributes
        self.evaluation = None  # Placeholder for an Evaluation object
        self.cv_evaluations = []  # List containing all evaluation objects in a cross-validation run

        # Set up run parameters attributes
        def copy_from_default_if_none(input_object_parameters, default_parameters):
            """
            Nested function copying default parameters defined inside the Runner class into the instantiated Runner
            object for input parameters that were not specified during construction.
            :param input_object_parameters: Parameters specified during object construction.
            :param default_parameters: Parameter defaults defined in the Runner class
            :return: Processed parameters
            """

            if input_object_parameters is None:
                object_parameters = default_parameters
            else:
                object_parameters = default_parameters.copy()
                object_parameters.update(input_object_parameters)
            return object_parameters

        self.training_parameters = copy_from_default_if_none(training_parameters, Runner._default_training_parameters)
        self.data_parameters = copy_from_default_if_none(data_parameters, Runner._default_data_parameters)
        self.compile_options = copy_from_default_if_none(compile_options, Runner._default_compile_options)

        # Other Attributes
        self._state = RunnerState.Initial  # Enum object state
        self.history = PrettyDict()  # Dictionary containing information about previous processing
    # endregion

    # region State-related Methods
    def advance_state(self):
        """
        Set the state attribute to the next enum state, by enum values order.
        :return:
        """
        all_states = RunnerState.all_states()
        current_index = all_states.index(self._state)
        if current_index < len(all_states)-1:
            self._state = all_states[current_index + 1]
    # endregion

    # region Data Loading Methods
    def load_data(self, read_kwargs=None):
        """
        Load the data from the given data directory.
        :param read_kwargs: Keyword arguments dictionary specifying options for the specific method that reads the data
        """
        print('Loading data')
        # Create Data Loader object based on the format of the data to be read
        loader = Loader.DataLoader.loader_factory(self.data_format, self.data_dir)
        # Load the data with the loader object
        self.data = loader.load_data(read_kwargs)

        # Create a Data object from the loaded data and store it
        # Assume working with numpy, could be changed to accommodate other libraries (pandas etc. if needed)
        self.data = Data.AntData(loader.to_numpy(self.data), loader.filenames, self)
        self._state = RunnerState.Loaded  # Advance Runner object state
    # endregion

    # region Preparation Methods
    """    
    Methods initializing Model and Evaluation objects before processing in a pipeline.
    """
    def initialize_model(self):
        """
        Initialize Model object.
        """
        # Initialize empty Model object
        self.model = model.model_factory(None, None, self)

        # Append model to a list for cross-validation purposes
        self.cv_models.append(self.model)

    def initialize_evaluation(self):
        """
        Initialize Evaluation object.
        """
        # Initialize evaluation object
        self.evaluation = evaluation.Evaluation(self)

        # Append evaluation object to a list for cross-validation purposes
        self.cv_evaluations.append(self.evaluation)
    # endregion

    # region Processing Methods
    def process(self, commands_list: list[tuple[Callable, list, dict, Optional[list[Enum]]]]):
        """
        Sequentially process the commands given in the pipeline.
        :param commands_list: List of 4-tuples of the format (method, list of method arguments, dictionary of
        keyword arguments, list of enum types selecting object instances to perform method on(None if irrelevant))
        """
        pipe = Pipeline(commands_list)  # Create Pipeline object

        # Process the methods given in the pipeline for the current Data, Model and Evaluation object instances stored
        # in the main Runner object
        pipe.process([self.data, self.model, self.evaluation])
        self.advance_state()  # Advance Runner object state

    def cross_validation_process(self, commands_list: list[tuple[Callable, list, dict, Optional[list[Enum]]]]):
        """
        Sequentially process the commands given in the pipeline over cross-validation data splits.
        :param commands_list: List of 4-tuples of the format (method, list of method arguments, dictionary of
        keyword arguments, list of enum types selecting object instances to perform method on(None if irrelevant))
        """
        pipe = Pipeline(commands_list)  # Create Pipeline object
        folds = list(range(self.training_parameters['number_of_folds']))
        for validation_fold_index in folds:  # Loop over cross-validation splits
            print('validation fold index: ' + str(validation_fold_index) + '\n')
            # Get list of training folds, each containing a list of indices of window samples
            training_window_indices_in_folds = [self.data.window_indices_in_folds[fold] for fold in folds if fold !=
                                                validation_fold_index]

            # Create training data object from the full data object according to the current split using those folds
            current_training_data = self.data.from_existing(training_window_indices_in_folds, Data.DataType.Training)

            # Get indices of window samples which are in the validation fold, contained within a list for further
            # processing
            validation_window_indices = [self.data.window_indices_in_folds[validation_fold_index]]

            # Create validation data object from the full data object according to the current split using those indices
            current_validation_data = self.data.from_existing(validation_window_indices, Data.DataType.Validation)

            self.initialize_model()  # Initialize model object for the current cross-validation split

            # Update Data objects stored in the model
            self.model.training_data = current_training_data
            self.model.validation_data = current_validation_data
            self.model.validation_fold_index = validation_fold_index

            self.initialize_evaluation()  # Initialize evaluation object for the current cross-validation split

            # Process the methods given in the pipeline for the current training and validation Data objects and the
            # Model and Evaluation object instances stored in the main Runner object
            pipe.process([current_training_data, current_validation_data, self.model, self.evaluation])
        self._state = RunnerState.CrossValidatedRun  # Advance Runner object state
    # endregion

    # region Static Methods
    @staticmethod
    def get_set_randomness(filename: str = 'pickled_random_state.pic'):
        """
        Load/save random state
        """
        try:
            # Try loading the file
            with open(filename, 'rb') as f:
                prng = pickle.load(f)

            # If file exists, get the saved random state from it and set the random module to it
            state = prng.get_state()
            np.random.set_state(state)
        except FileNotFoundError:
            # If file does not exist, get the random state and save it
            prng = np.random.RandomState()
            with open(filename, 'wb') as f:
                pickle.dump(prng, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            raise error
    # endregion
# endregion


# region PrettyDict Class
class PrettyDict(dict):
    """
    Class extending the base dictionary class for a nicer representation of the dictionary when printing to the
    command line.
    """
    def __repr__(self):
        """
        Build a item-per-line string representation of the dictionary
        :return: A string containing the string representation of the dictionary
        """
        max_item_index = len(self)-1
        repr_out = '\n{'
        for num, kv in enumerate(self.items()):
            if num < max_item_index:
                extra = '\n'
            else:
                extra = '}\n'
            repr_out = repr_out + '{} : {}'.format(kv[0], kv[1]) + extra
        return repr_out
# endregion
