import pandas as pd
import re
import os
import glob
import numpy as np
import math

# import timeit
# starttime = timeit.default_timer()
# print("scipy dilate time:", timeit.default_timer() - starttime)

# region Globals
DATA_DIR = r'\\phys-guru-cs\ants\Aviram\Marking behavior data'
MARKING_TIME = 0.1  # typical marking time in sec
fps = 50
# endregion Globals

# region DataLoader Class and child classes
# region Base Loader Class
class DataLoader:

    # region Constructor
    def __init__(self, dir_path):
        self.path = dir_path
        self.filenames = None
        self.error = None
    # endregion

    # region Methods
    def get_file_list(self, extension):
        self.filenames = glob.glob(os.path.join(self.path, '**', '*.' + extension), recursive=True)

    def load_data(self, read_kwargs):
        data = []
        for fname in self.filenames:
            try:
                read_array, empty_flag = self.read_file(fname, read_kwargs)
                if not empty_flag:
                    data.append(read_array)
            except self.error:
                fname_end = re.split(r'\\', fname)
                fname_end = fname_end[-2] + '\\' + fname_end[-1]
                print('file {} is empty and has been skipped'.format(fname_end))
            except Exception as error:
                raise error
        return data
    # endregion

    # region Abstract Methods
    def to_numpy(self, data):
        return data  # assigned to make sure the editor correctly evaluates the output type

    def read_file(self, fname, read_kwargs):
        return fname
    # endregion
# endregion


# region csvLoader
class csvLoader(DataLoader):
    # region Constructor
    def __init__(self, dir_path):
        super().__init__(dir_path)
        self.get_file_list('csv')
        self.error = pd.errors.ParserError
    # endregion

    def to_numpy(self, data):
        numpy_data = [d.to_numpy() for d in data]
        return numpy_data

    def load_data(self, read_kwargs=None):
        # change immutable default to needed mutable default value
        if read_kwargs is None:
            read_kwargs = {}
        return super().load_data(read_kwargs)

    # region Overriding Methods
    def read_file(self, fname, read_kwargs):
        read_array = pd.read_csv(fname, **read_kwargs)
        empty_flag = False
        if read_array.empty:
            empty_flag = True
        return read_array, empty_flag
    # endregion

# endregion
# endregion

def smear_labels(labels):
    # currently only rectangular window, add other window types if needed
    # disregarding 1,2 labels - smear everything into 1 class
    window_size = math.ceil(MARKING_TIME*fps)
    if not window_size % 2:
        window_size += 1
    for sequence in labels:
        label_inds = np.nonzero(sequence)[0]
        if label_inds.size:
            i_minus = label_inds-(window_size-1)/2
            i_plus = label_inds+(window_size-1)/2
            inds_in_window = [np.arange(max(im, 0), min(ip+1, sequence.shape[0]), dtype=int) for
                              im, ip in zip(i_minus, i_plus)]
            inds_in_window = np.unique(np.concatenate(inds_in_window))
            sequence[inds_in_window] = 1
    return labels

def separate_circular_features(features, circular_indices):
    for sequence in features:
        rads = math.radians(sequence[:,circular_indices])
        insert(np.stack(math.cos(rads), math.sin(rads)),

def cut_array_into_windows(features, labels, window_size, stride):
    # Cut data and labels into overlapping windows of size window_size, with stride stride.
    # Zero-pad samples with size<window_size

    stack_list = []
    for feature_sequence, label_sequence in zip(features, labels):
        length = feature_sequence.shape[0]
        last_full_window_ind = max(math.floor((length-window_size)/stride+1), 1)
        sequence_list = []
        for i in range(0, length, stride):
            sequence_list.append(np.c_[feature_sequence[i: i + window_size, :], label_sequence[i: i + window_size]])

        sequence_list = sequence_list[:last_full_window_ind]  # remove last part to make sure sizes are the same
        if last_full_window_ind == 1:  # Zero-pad samples with size<window_size
            sequence_list[0] = np.pad(sequence_list[0], ((0, window_size-sequence_list[0].shape[0]), (0, 0)))
        stack_list.append(np.stack(sequence_list, axis=0))
    return np.concatenate(stack_list, axis=0)  # concatenate all samples and return them as a 3D numpy array

# TODO: stratify data?
# TODO: augmentation - rotate X,Y? what else?
# TODO: circular variables -> cos and sin features
# TODO: arrange data in big ol' Tensor (batch size should be determined in the model training part)







#  starttime = timeit.default_timer()
loader = csvLoader(DATA_DIR)
ant_data = loader.load_data() #{'header': None}
ant_data = loader.to_numpy(ant_data)
features = [mat[:, 1:-1] for mat in ant_data]  #remove name, frame and labels
labels = [mat[:, -1] for mat in ant_data]  # 3 classes in the data
window_size = 500
stride = 150
smear_labels(labels)
cut_array_into_windows(features, labels, window_size, stride)




