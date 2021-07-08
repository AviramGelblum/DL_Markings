import pandas as pd
import re
import os
import glob


# import timeit
# starttime = timeit.default_timer()

# print("scipy dilate time:", timeit.default_timer() - starttime)



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