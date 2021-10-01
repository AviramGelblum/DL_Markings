from __future__ import annotations

from UNet1D import UNet1D
from Pipeline import IProcessable
import evaluation

import numpy as np
import copy
from typing import Optional, Union, TYPE_CHECKING

import tensorflow.keras.backend as K_backend
import tensorflow.keras.callbacks as K_callbacks

if TYPE_CHECKING:
    # Always False, allows compile-time type checking of variables of classes whose importing would raise run-time
    # circular import errors. Could simply use if False here, but this way provides a better understanding of what
    # this does and Pycharm doesn't raise a warning regarding unreachable code.
    from Runner import Runner
    from tensorflow import Tensor as tf_tensor
    # noinspection PyProtectedMember
    from tensorflow._api.v2.data import Dataset as tf_dataset


# region Model Factory Function
def model_factory(model_type, input_size, runner, kwargs=None):
    """
    Factory function generating Model objects
    :param model_type: Type of underlying model to generate
    :param input_size: Tuple defining dimensions of the input to the model
    :param runner: Parent Runner object
    :param kwargs: Dictionary containing keyword arguments specifying options for the underlying model object
    constructor
    :return: Instantiated Model object containing a built model of type model_type
    """
    # Change immutable default to needed mutable default value
    if kwargs is None:
        kwargs = {}

    # Dictionary associating allowed model type strings to their respective model classes implementing them
    # noinspection SpellCheckingInspection
    model_class_dict = {'unet': UNet1D, 'None': None}
    if model_type in model_class_dict:
        model_class = model_class_dict[model_type]  # If the model type is allowed, get the underlying model class
        # Instantiate a Model object receiving an instantiated underlying model object as input, and return it
        return Model(model_class(input_size, **kwargs), runner)
    elif model_type is None:
        # Empty Model object instantiation option
        return Model(None, runner)
    raise KeyError(model_type + ' is not a valid model type.')
# endregion


# region Model Class
class Model(IProcessable):
    # noinspection SpellCheckingInspection
    """
        A wrapper class for Keras deep learning models processing, implementing the IProcessable interface allowing
        instances to be processed in a Pipeline.
    """

    # region Constructor
    def __init__(self, model: Optional[UNet1D], runner: Runner):
        """
        Constructor method for the Model class.
        :param model: Underlying Keras model, built but not yet compiled
        :param runner: Parent Runner object
        """
        # Keras Model Attributes
        self.built_model = model
        self.model_type = runner.model_type  # Underlying Keras model type
        self.scheduler = K_callbacks.LearningRateScheduler(self.lrscheduler)  # Keras learning rate scheduler during-training callback

        # Parameters Attributes
        # See Runner.Runner [region Basic Parameters and Defaults] for explanation of each parameter
        self.training_parameters = runner.training_parameters
        self.compile_options = runner.compile_options

        # Data Attributes
        self.training_data = None  # Placeholder for Data object containing the data in the training folds
        self.validation_data = None  # Placeholder for Data object containing the data in the validation fold
        self.validation_fold_index = None  # Placeholder for the data split configuration used in the current model

        # Prediction Attributes
        # Placeholder for post-training model predictions on validation data
        self.validation_predicted_probabilities = None

        # Other Attributes
        self.runner = runner
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
        return True  # No type verification for Model objects - always process
    # endregion

    # region Processing Methods
    def create(self):
        """
        Create and build the underlying Keras model.
        """
        # Check that data is provided, so that the input size for the keras model can be calculated
        if self.training_data is None or self.validation_data is None:
            raise ValueError('training/validation data was not provided during model creation')
        input_size = self.training_data.windowed_features.shape[1:]

        # Build underlying Keras model via the Model object model_factory
        self.built_model = model_factory(self.model_type, input_size, self.runner).built_model
        self.history['model_created'] = True  # Update history

    def compile(self):
        """
        Compile the underlying Keras model.
        """
        K_backend.clear_session()  # Before running the model, reset all states generated by Keras.

        # Construct optimizer object
        kw_in = copy.copy(self.compile_options)
        kw_in['optimizer'] = kw_in['optimizer'](kw_in['learning_rate'])
        del kw_in['learning_rate']

        self.built_model.model.compile(**kw_in)  # Compile model
        self.history['model_compiled'] = True  # Update history

    def train(self, input_callbacks: Optional[list[K_callbacks.Callback]] = None):
        """
        Train the underlying Keras model.
        :param input_callbacks: List of tensorflow.keras.callbacks objects to be called during model training. See
        https://keras.io/api/callbacks/, https://keras.io/guides/writing_your_own_callbacks/ for more information on
        callbacks.
        """
        # Add optional input callbacks to default callbacks
        callbacks = [self.scheduler, evaluation.DilatedOnTrainingEvaluation
                     (self.built_model.model, self.validation_data, tolerance=self.training_parameters['prediction_window_size'],
                      probability_threshold=self.training_parameters['probability_threshold'])]
        if input_callbacks:
            callbacks.extend(input_callbacks)

        # Train underlying Keras model
        self.built_model.model.fit(x=self.training_data.windowed_features,
                                   y=self.training_data.windowed_labels,
                                   batch_size=self.training_parameters['batch_size'],
                                   epochs=self.training_parameters['n_epoch'],
                                   validation_data=(self.validation_data.windowed_features,
                                                    self.validation_data.windowed_labels),
                                   callbacks=callbacks,
                                   shuffle=True)

        # Delete training data to save memory
        if self.training_parameters['discard_data']:
            self.training_data.set_to_null()
        self.history['model_trained'] = True  # Update history

    def predict(self, features: Optional[Union[tf_tensor, tf_dataset, np.array]] = None):
        """
        Use trained model to predict label sequences from given feature sequences.
        :param features: Optional input features. Must obey the type constraints for the tf.keras.model.predict method
        as specified in https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict, and follow the dimensions
        the model expects. If not specified, the method predicts probabilities for the validation data.
        :return: If input features are given, return the prediction of the model as numpy array sequences of predicted
        probabilities, else the method has no output arguments.
        """
        if features is None:
            # Predict label sequences for validation data
            self.validation_predicted_probabilities = np.array(self.built_model.model(self.validation_data.windowed_features))

            if isinstance(self.runner.evaluation, evaluation.Evaluation):
                # Set/update resulting probabilities in the corresponding Evaluation object
                self.runner.evaluation.predicted_probabilities = self.validation_predicted_probabilities

            # Delete validation data to save memory
            if self.training_parameters['discard_data']:
                self.validation_data.set_to_null()
        else:
            # Return model prediction for input features, if provided
            return np.array(self.built_model.model(features))
    # endregion

    # region Training Callback Methods
    # noinspection SpellCheckingInspection
    def lrscheduler(self, epoch, learning_rate):
        """
        Scheduler function for Keras LearningRateScheduler callback. Decreases learning rate as a function of the
        current epoch number.
        :param epoch: current epoch (Tensorflow internal variable)
        :param learning_rate: current learning rate (Tensorflow internal variable, taken from the optimizer object)
        :return: adjusted learning rate
        """
        if epoch < self.training_parameters['n_epoch']*0.75:
            lr = learning_rate
        elif epoch < self.training_parameters['n_epoch']*0.9:
            lr = learning_rate/10
        else:
            lr = learning_rate/100
        return lr
    # endregion
# endregion
