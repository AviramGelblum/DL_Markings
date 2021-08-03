from tensorflow.keras.utils import Sequence
import numpy as np
import preprocessing as pp


class DataGenerator(Sequence):
    def __init__(self, data, feature_types, batch_size=32, augment_factor=None, shuffle=True):
        """
        initialization method for DataGenerator class.
        :param data: train or validation dictionary with three keys:
        'features' - containing a numpy array of shape (number of examples, sequence length, number of features)
        'labels - containing a numpy array of shape (number of examples, sequence length)
        'association' - containing a numpy array of shape (number of examples)
        :param feature_types: dictionary referencing the indices of Circular, Spatial and Velocity related feature
        types
        :param batch_size: size of batch: int.
        :param augment: boolean. Should be set to true if we want to augment. If we do not augment then using the
        data generator gives us no added value in terms of memory since we preload and preprocess the data. If we do
        augment on generation then we save on memory but pay in repeated augmentation time after each epoch. Also,
        we get different random rotations for each epoch. Must be False if we are generating validation data.
        :param shuffle: shuffle the order the network sees the examples.
        """

        self.batch_size = batch_size
        self.data = data
        self.shape = data['features'].shape

        self.augment = augment_factor
        if self.augment:
            self.real_example_size = self.batch_size // self.augment  # 8 kinds of augmentations per point (2 rotations,
            # 2 flips per rotation)
            num_examples_to_fill = self.batch_size % self.augment
            self.examples_used_per_batch = self.real_example_size + num_examples_to_fill

        self.feature_types = feature_types
        self.shuffle = shuffle

        self.indexes = None  # defined in on_epoch_end
        self.on_epoch_end()  # initialize shuffled example indices

    def __len__(self):
        if self.augment:
            return int(np.floor(self.shape[0] / self.examples_used_per_batch))
        else:
            return int(np.floor(self.shape[0] / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        if self.augment:
            indexes = self.indexes[index * self.examples_used_per_batch:
                                   (index + 1) * self.examples_used_per_batch]
            additional_indexes = indexes[self.real_example_size:]
            indexes = indexes[:self.real_example_size]
            partial_data = {k: v[indexes] for k, v in self.data.items()}

            pp.augment_random_rotations(partial_data, self.feature_types['Circular'], self.feature_types['Spatial'])
            pp.augment_xy_flip(partial_data, self.feature_types['Circular'], self.feature_types['Spatial'])

            partial_data['features'] = np.concatenate([partial_data['features'], self.data['features'][additional_indexes]])
            partial_data['labels'] = np.concatenate([partial_data['labels'], self.data['labels'][additional_indexes]])

            pp.separate_circular_features(partial_data, self.feature_types['Circular'])

            features = partial_data['features']
            labels = partial_data['labels']
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            features = self.data['features'][indexes]
            labels = self.data['labels'][indexes]

        return features, labels

    def on_epoch_end(self):
        self.indexes = np.arange(self.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)
