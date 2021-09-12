import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras as keras
import math


class UNet1D:
    __default_dropout = [None, None, 0.5, 0.5]
    __default_maxpool = [3, 3, 3, None]

    # sequence-to-sequence 1D UNet model
    def __init__(self, input_size, depth=4, kernel_size=3, initial_num_filters=32, dropout=None,
                 maxpool=None, BatchNorm=True):

        self.kernel_size = kernel_size
        self.depth = depth

        self.__validate_input_size(input_size)
        UNet1D.__validate_initial_num_filters(initial_num_filters)
        self.input_size = input_size

        if dropout is None:
            self.dropout = UNet1D.__default_dropout
        elif isinstance(dropout, str) and dropout.lower() == 'none':
            self.dropout = self.__expand_none_to_depth(None)

        if maxpool is None:
            self.maxpool = UNet1D.__default_maxpool
        elif isinstance(dropout, str) and maxpool.lower() == 'none':
            self.maxpool = self.__expand_none_to_depth(None)

        self.BatchNorm = BatchNorm
        self.num_filters = self.__calculate_filters(initial_num_filters)
        self.current_layer = None
        self.model = self.__build()

    @staticmethod
    def __validate_initial_num_filters(initial_num_filters):
        log_num_filters = math.log2(initial_num_filters)
        if log_num_filters != math.floor(log_num_filters):
            raise ModelInputError('initial_num_filters must be a power of 2')

    def __validate_input_size(self, input_size):
        if input_size[0] % self.kernel_size**self.depth:
            raise ModelInputError('Input must be divisible by ' + str(self.kernel_size ** self.depth))

    def __calculate_filters(self, initial_num_filters):
        initial_filter_power = int(math.log2(initial_num_filters))
        filters = [2**filter_power for filter_power in
                   range(initial_filter_power, self.depth+initial_filter_power)]
        return filters

    def __expand_none_to_depth(self, property_input):
        if property_input is None:
            property_input = [property_input]*(self.depth+1)
        return property_input

    def __build(self):
        inputs = keras.Input(self.input_size)
        self.current_layer = inputs
        residuals = []
        for num_filters, dropout, maxpool in zip(self.num_filters, self.dropout, self.maxpool):
            block = SepConvLayerBlock(self.current_layer, num_filters, self.kernel_size, activation='relu',
                                      BatchNorm=self.BatchNorm, dropout=None, max_pool_size=None)
            self.current_layer = block.create_block()
            del block

            block = SepConvLayerBlock(self.current_layer, num_filters, self.kernel_size, activation='relu',
                                      BatchNorm=self.BatchNorm, dropout=dropout, max_pool_size=maxpool)
            if maxpool:
                self.current_layer, residual = block.create_block()
                residuals.append(residual)
            else:
                self.current_layer = block.create_block()
            del block

        residuals.reverse()

        for num_filters, residual in zip(self.num_filters[self.depth-2::-1], residuals):
            block = TransConvLayerBlock(self.current_layer, num_filters, self.kernel_size, self.kernel_size,
                                        activation='relu',  BatchNorm=False, dropout=None, max_pool_size=None)
            self.current_layer = block.create_block()
            del block

            self.current_layer = layers.concatenate([residual, self.current_layer], axis=2)

            block = SepConvLayerBlock(self.current_layer, num_filters, self.kernel_size, activation='relu',
                                      BatchNorm=self.BatchNorm, dropout=None, max_pool_size=None)
            self.current_layer = block.create_block()
            del block

            block = SepConvLayerBlock(self.current_layer, num_filters, self.kernel_size, activation='relu',
                                      BatchNorm=self.BatchNorm, dropout=None, max_pool_size=None)
            self.current_layer = block.create_block()
            del block
        self.current_layer = layers.Conv1D(1, 1, activation='sigmoid', padding="same")(self.current_layer)

        model = models.Model(inputs, self.current_layer)
        return model


class LayerBlock:
    def __init__(self,current_layer, filters, kernel_size, activation, BatchNorm, dropout, max_pool_size):
        self.filters = filters
        self.BatchNorm = BatchNorm
        self.dropout = dropout
        self.max_pool_size = max_pool_size
        self.current_layer = current_layer
        self.kernel_size = kernel_size
        self.activation = activation

    def create_block(self):
        current_layer = self.current_layer
        if self.BatchNorm:
            current_layer = layers.BatchNormalization()(current_layer)
        current_layer = layers.Activation(self.activation)(current_layer)
        if self.dropout:
            current_layer = layers.Dropout(self.dropout)(current_layer)
        if self.max_pool_size:
            residual = current_layer
            current_layer = layers.MaxPooling1D(pool_size=self.max_pool_size)(current_layer)
            return current_layer, residual
        return current_layer


class SepConvLayerBlock(LayerBlock):
    def __init__(self, current_layer, filters, kernel_size, activation='relu', BatchNorm=True, dropout=None,
                 max_pool_size=None):
        super().__init__(current_layer, filters, kernel_size, activation, BatchNorm, dropout, max_pool_size)

    def create_block(self):
        self.current_layer = layers.SeparableConv1D(self.filters, self.kernel_size, activation=self.activation,
                                                    padding="same", kernel_initializer='he_normal')(self.current_layer)
        if self.max_pool_size:
            current_layer, residual = super().create_block()
            return current_layer, residual
        else:
            current_layer = super().create_block()
            return current_layer


class TransConvLayerBlock(LayerBlock):
    def __init__(self,current_layer, filters, kernel_size, strides, activation='relu', BatchNorm=True, dropout=None,
                 max_pool_size=None):
        super().__init__(current_layer, filters, kernel_size, activation, BatchNorm, dropout, max_pool_size)
        self.strides = strides

    def create_block(self):
        self.current_layer = layers.Conv1DTranspose(self.filters, self.kernel_size, activation=self.activation,
                                                    strides=self.strides, kernel_initializer='he_normal')(self.current_layer)
        current_layer = super().create_block()
        return current_layer


class ModelInputError(Exception):
    pass
