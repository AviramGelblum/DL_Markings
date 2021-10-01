import math
from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.keras import Input
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


# region UNet1D Class
class UNet1D:
    # noinspection SpellCheckingInspection
    """
        Keras implementation of a sequence-to-sequence deep 1D convolutional neural network based on the UNet
        spatial contraction-expansion architecture as described in U-Net: Convolutional Networks for Biomedical Image Segmentation
         (Ronneberger, Fischer and Brox 2015). See original article at https://arxiv.org/abs/1505.04597.
        """
    # region default parameters
    _default_dropout = [None, None, 0.5, 0.5]  # Dropout proportion at each level of the descending part of the network
    _default_max_pool = [3, 3, 3, None]  # Max pooling spatial contraction factor at each level of the descending part of the network
    # endregion

    def __init__(self, input_size: tuple[int, int], depth: int = 4, kernel_size: int = 3, initial_num_filters: int = 32,
                 dropout: Union[list[Optional[int]], str] = None, max_pool: list[Optional[int]] = None,
                 BatchNorm: bool = True):
        """
        Constructor method for the UNet1D model class
        :param input_size: Two-element tuple describing feature input dimensions (sequence segment length,
        number of features)
        :param depth: Number of levels of network
        :param kernel_size: 1D convolution kernel size
        :param initial_num_filters: Number of filters at the first level. Must be a power of 2.
        :param dropout: Dropout proportion at each level of the descending part of the network
        :param max_pool: Max pooling spatial contraction factor at each level of the descending part of the network
        :param BatchNorm: Boolean controlling whether Batch Normalization is performed on the input before entering
        it into the activation function
        """
        # Model Architecture Parameters
        self.kernel_size = kernel_size
        self.depth = depth
        self.BatchNorm = BatchNorm

        # Validate input sequence length, to make sure spatial contraction results in an integer sequence length at every level of the network.
        self._validate_input_size(input_size)
        self.input_size = input_size

        # Validate initial number of filters is a power of 2, to make sure filter number increase results in an integer number of filters at every level of the network
        UNet1D._validate_initial_num_filters(initial_num_filters)
        # Calculate number of filters for each level, given initial number of filters
        self.num_filters: list = self._calculate_filters(initial_num_filters)

        # Set Dropout proportion at each level
        if dropout is None:
            # Default dropout scheme was not overridden by optional keyword argument
            self.dropout = UNet1D._default_dropout
        elif isinstance(dropout, str) and dropout.lower() == 'none':
            # No dropout, create a list of None of length equal to the depth of the network.
            self.dropout = [None]*(self.depth+1)

        if max_pool is None:
            # Default max pooling scheme was not overridden by optional keyword argument
            self.max_pool = UNet1D._default_max_pool

        # Model Attributes
        self.current_layer = None  # Initialization of the model layers, on top of which we build the model by adding layers sequentially
        self.model = self._build()  # Build the model architecture

    # region Private Methods
    # region Input Validation Methods
    @staticmethod
    def _validate_initial_num_filters(initial_num_filters):
        """
        Validate that the number of filters in the first level is a power of 2
        :param initial_num_filters: Input number of filters in the first level.
        """
        log_num_filters = math.log2(initial_num_filters)
        if log_num_filters != math.floor(log_num_filters):
            raise ModelInputError('initial_num_filters must be a power of 2')

    def _validate_input_size(self, input_size):
        """
        Validate that the input sequence length is divisible (without remnant) by the maximum contraction factor
        kernel_size^depth, which occurs at the deepest level of the network
        :param input_size: Input sequence length.
        """
        if input_size[0] % self.kernel_size**self.depth:
            raise ModelInputError('Input must be divisible by ' + str(self.kernel_size ** self.depth))
    # endregion

    # region Other Methods
    def _calculate_filters(self, initial_num_filters):
        """
        Calculate number of filters for each level, given number of filters at the first level of the model.
        :param initial_num_filters: Input number of filters in the first level.
        :return: A list containing number of filters at each level of the model.
        """
        initial_filter_power = int(math.log2(initial_num_filters))
        filters = [2**filter_power for filter_power in
                   range(initial_filter_power, self.depth+initial_filter_power)]
        return filters

    def _build(self):
        """
        Build Keras model architecture.
        :return: Built Keras U-Net-inspired model
        """
        # Create input layer
        inputs = Input(self.input_size)
        self.current_layer = inputs
        residuals = []
        for num_filters, dropout, max_pool in zip(self.num_filters, self.dropout, self.max_pool):
            # Build descending/contracting pathway of the network. Each iteration of the loop builds a single level
            # composed of 2 separable 1D convolution layers and a 1D max-pooling layer contracting the spatial size (
            # while the number of filters increases exponentially), except in the deepest level of the network.

            # Create the 1st separable 1D convolution layer in the level
            block = SepConvLayerBlock(self.current_layer, num_filters, self.kernel_size, activation='relu',
                                      BatchNorm=self.BatchNorm, dropout=None, max_pool_size=None)
            self.current_layer = block.create_block()
            del block

            # Create the 2nd 1D convolution layer in the level, with max pooling
            block = SepConvLayerBlock(self.current_layer, num_filters, self.kernel_size, activation='relu',
                                      BatchNorm=self.BatchNorm, dropout=dropout, max_pool_size=max_pool)
            if max_pool:
                self.current_layer, residual = block.create_block()
                # Output from the last layer in each  level of the descending path is used as additional input (through
                # concatenation) to the first layer in the corresponding level of the ascending path
                # Save residuals for this future lateral horizontal input transfer between descending and ascending
                # sections of the network.
                residuals.append(residual)
            else:
                # No lateral horizontal input transfer at the deepest level.
                self.current_layer = block.create_block()
            del block

        # Order residuals from the bottom-up, to be used as the ascending path is built from the deepest level up
        residuals.reverse()

        for num_filters, residual in zip(self.num_filters[self.depth-2::-1], residuals):
            # Build ascending/expanding pathway of the network. Each iteration of the loop builds a single level
            # composed of a single transpose convolution expanding the spatial size back, followed by 2 separable 1D
            # convolution layers.

            # Create a 1D transpose convolution layer as the spatial expansion operation of choice.
            block = TransConvLayerBlock(self.current_layer, num_filters, self.kernel_size, self.kernel_size,
                                        activation='relu',  BatchNorm=False, dropout=None, max_pool_size=None)
            self.current_layer = block.create_block()
            del block

            # Concatenate output data from the last layer in the corresponding descending path level to the input of
            # the first Separable 1D convolution layer in the current ascending path level
            self.current_layer = layers.concatenate([residual, self.current_layer], axis=2)

            # Create the 1st separable 1D convolution layer in the level
            block = SepConvLayerBlock(self.current_layer, num_filters, self.kernel_size, activation='relu',
                                      BatchNorm=self.BatchNorm, dropout=None, max_pool_size=None)
            self.current_layer = block.create_block()
            del block

            # Create the 2nd separable 1D convolution layer in the level
            block = SepConvLayerBlock(self.current_layer, num_filters, self.kernel_size, activation='relu',
                                      BatchNorm=self.BatchNorm, dropout=None, max_pool_size=None)
            self.current_layer = block.create_block()
            del block

        # Last layer of the network is a simple 1D pointwise convolution with a single filter, outputting a 1D
        # probability sequence with spatial dimensions equal to that of the original input sequence
        self.current_layer = layers.Conv1D(1, 1, activation='sigmoid', padding="same")(self.current_layer)

        # Create Keras model from input and layer architecture
        model = models.Model(inputs, self.current_layer)
        return model
    # endregion
    # endregion
# endregion


# region LayerBlock Classes
# region LayerBlock Abstract Base Class
class LayerBlock(metaclass=ABCMeta):
    """
    Abstract base class for creating a Keras Layer Block including sequential optional Batch Normalization, activation,
    optional dropout and optional max pooling operation.
    """
    # region Constructor
    @abstractmethod
    def __init__(self, current_layer: KerasTensor, filters: int, kernel_size: int, activation: str, BatchNorm: bool,
                 dropout: Optional[float], max_pool_size: Optional[int]):
        """
        Abstract base constructor method for LayerBlock objects. This abstract method is invoked by the
        constructors of concrete subclasses of LayerBlock.
        :param current_layer: A KerasTensor object accumulating the layers throughout model building
        :param filters: Number of filters to be used in the convolution procedure.
        :param kernel_size: Size of filters to be used in the convolution procedure.
        :param activation: String specifying type of activation.
        :param BatchNorm: Boolean signifying whether to perform batch normalization before activation or not
        :param dropout: Proportion of nodes to drop out in the descending path of the network during training,
        after processing the convolved data through the activation function. If None, then no dropout layer is added.
        :param max_pool_size: Spatial contraction factor applied in the descending path of the network through max
        pooling operation.
        """
        # Convolution Parameters Attributes
        self.filters = filters
        self.kernel_size = kernel_size

        # Operations Attributes
        self.BatchNorm = BatchNorm
        self.dropout = dropout
        self.max_pool_size = max_pool_size
        self.activation = activation

        # KerasTensor Model Layer Architecture Object Attribute
        self.current_layer = current_layer
    # endregion

    # region Abstract Methods
    @abstractmethod
    def create_block(self):
        """
        Abstract method implementing partial building of a processing block with multiple possible operations.
        :return: current_layer - KerasTensor object containing the current layer in the mode architecture.
                 residual - If max pooling is performed, also return the current_layer object before the max pooling
                 operation is performed.
        """
        current_layer = self.current_layer
        # Batch Normalization
        if self.BatchNorm:
            current_layer = layers.BatchNormalization()(current_layer)

        # Activation
        current_layer = layers.Activation(self.activation)(current_layer)

        # Dropout
        if self.dropout:
            current_layer = layers.Dropout(self.dropout)(current_layer)

        # Max pooling
        if self.max_pool_size:
            residual = current_layer  # Save residual for later descending-ascending path lateral feature map transfer
            current_layer = layers.MaxPooling1D(pool_size=self.max_pool_size)(current_layer)
            return current_layer, residual
        return current_layer
    # endregion
# endregion


# region SepConvLayerBlock Concrete Subclass (LayerBlock)
class SepConvLayerBlock(LayerBlock):
    """
    Concrete subclass inheriting from LayerBlock, implementing a processing block with separable 1D convolution
    operation.
    """
    # region Constructor
    def __init__(self, current_layer, filters, kernel_size, activation='relu', BatchNorm=True, dropout=None,
                 max_pool_size=None):
        """
        Constructor method for SepConvLayerBlock class. See parent LayerBlock class's docstring for explanations about
        input variables.
        """
        super().__init__(current_layer, filters, kernel_size, activation, BatchNorm, dropout, max_pool_size)
    # endregion

    # region Methods
    def create_block(self):
        """
        Build the processing block.
        :return: current_layer - KerasTensor object containing the current layer in the model architecture.
                 residual - If max pooling is performed, also return the current_layer object before the max pooling
                 operation is performed.
        """
        # Separable 1D convolution
        self.current_layer = layers.SeparableConv1D(self.filters, self.kernel_size, activation=self.activation,
                                                    padding="same", kernel_initializer='he_normal')(self.current_layer)

        # Other processing options implemented in the parent class' overridden method.
        if self.max_pool_size:
            current_layer, residual = super().create_block()
            return current_layer, residual
        else:
            current_layer = super().create_block()
            return current_layer
    # endregion
# endregion


# region TransConvLayerBlock Concrete Subclass (LayerBlock)
class TransConvLayerBlock(LayerBlock):
    """
    Concrete subclass inheriting from LayerBlock, implementing a processing block with 1D transpose convolution
    operation.
    """
    # region Constructor
    def __init__(self, current_layer, filters, kernel_size, strides, activation='relu', BatchNorm=True, dropout=None,
                 max_pool_size=None):
        """
        Constructor method for TransConvLayerBlock class. See parent LayerBlock class's docstring for explanations about
        input variables.
        :param strides: Stride length of the transpose 1D convolution operation. This parameter determines spatial
        expansion applied in the ascending path of the network.
        """
        super().__init__(current_layer, filters, kernel_size, activation, BatchNorm, dropout, max_pool_size)
        self.strides = strides
    # endregion

    # region Methods
    def create_block(self):
        """
        Build the processing block.
        :return: KerasTensor object containing the current layer in the model architecture.
        """
        # 1D transpose convolution
        self.current_layer = layers.Conv1DTranspose(self.filters, self.kernel_size, activation=self.activation,
                                                    strides=self.strides, kernel_initializer='he_normal')(self.current_layer)

        # Other processing options implemented in the parent class' overridden method.
        current_layer = super().create_block()
        return current_layer
    # endregion
# endregion
# endregion


# region Exceptions
class ModelInputError(Exception):
    """
    Custom exception raised when input model parameters (number of filters, size of input) do not conform to the
    imposed constraints of the model.
    """
    pass
# endregion
