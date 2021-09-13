import tensorflow.keras.backend as Kbackend
import tensorflow.keras.callbacks as Kcallbacks
from UNet1D import UNet1D
from Pipeline import Iprocessable
import evaluation
import numpy as np

# model_parameters = {'dilation_window_size': math.ceil(pp.MARKING_TIME * pp.fps), 'stride': int(1.25 * pp.fps),
#                     'sample_window_size': pp.divisible_window_size(6 * pp.fps, 81),
#                     'prediction_window_size': math.ceil(pp.MARKING_TIME * pp.fps)*2}
# training_parameters = {'learning_rate': 0.0005, 'n_epoch': 7, 'batch_size': 32, 'probability_threshold': 0.25,
#                        'is_data_from_generator': False, 'focal_loss_alpha': 0.25}


def model_factory(model_type, input_size, runner, kwargs=None):
    if kwargs is None:
        kwargs = {}

    model_class_dict = {'unet': UNet1D}
    if model_type in model_class_dict:
        model_class = model_class_dict[model_type]
        return Model(model_class(input_size, **kwargs), runner)
    else:
        raise KeyError(model_type + ' is not a valid model type.')


class Model(Iprocessable):
    def __init__(self, model, runner):
        self.runner = runner
        self.built_model = model
        self.training_parameters = runner.training_parameters
        self.training_data = None
        self.validation_data = None
        self.scheduler = Kcallbacks.LearningRateScheduler(self.lrscheduler)
        self.history = runner.history
        self.validation_predicted_probabilities = None
        self.validation_fold_index = None

    def verify_type(self, instance_types):
        return True

    def compile(self, kwargs):
        Kbackend.clear_session()
        self.built_model.model.compile(**kwargs)
        self.history['model_compiled'] = True

    def train(self, callbacks=None):
        batch_size = self.training_parameters['batch_size']
        self.built_model.model.fit(x=self.training_data.windowed_features,
                                   y=self.training_data.windowed_labels,
                                   batch_size=batch_size,
                                   epochs=self.training_parameters['n_epoch'],
                                   validation_data=(self.validation_data.windowed_features,
                                                    self.validation_data.windowed_labels),
                                   callbacks=[self.scheduler,
                                              evaluation.DilatedOnTrainingEvaluation
                                              (self.built_model.model, self.validation_data,
                                               tolerance=self.training_parameters['prediction_window_size'],
                                               probability_threshold=self.training_parameters['probability_threshold'])],
                                   shuffle=True)
        self.history['model_trained'] = True

    def predict(self, features=None):
        if features is None:
            self.validation_predicted_probabilities = np.array(self.built_model.model(self.validation_data.windowed_features))
            if isinstance(self.runner.evaluation, evaluation.Evaluation):
                self.runner.evaluation.predicted_probabilities = self.validation_predicted_probabilities
        else:
            return np.array(self.built_model.model(features))

    def lrscheduler(self, epoch, learning_rate):
        # scheduler function for LearningRateScheduler callback
        if epoch < self.training_parameters['n_epoch']*0.75:
            lr = learning_rate
        elif epoch < self.training_parameters['n_epoch']*0.9:
            lr = learning_rate/10
        else:
            lr = learning_rate/100
        return lr









