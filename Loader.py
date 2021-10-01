import pandas as pd
import re
import os
import glob
from abc import ABCMeta, abstractmethod
from typing import Optional


# region DataLoader Abstract Base Class
class DataLoader(metaclass=ABCMeta):
    """
    Abstract base class for data loading
    """
    # region Factory Static Method
    @staticmethod
    def loader_factory(data_format: str, dir_path: str):
        """
        Factory method instantiating a child class of DataLoader associated with the input data format
        :param data_format: Format of the data to be loaded
        :param dir_path: Path of the directory containing the data (or sub-folders containing the data)
        :return: An instantiation of the child class associated with the input data format
        """
        loader_class_dict = {'csv': csvLoader}  # Possible format-loader classes dictionary
        if data_format in loader_class_dict:
            loader_class = loader_class_dict[data_format]  # Get relevant child loader class
            return loader_class(dir_path)  # Return an instance of the child loader class
        else:
            raise KeyError(data_format + ' is not a valid data format.')
    # endregion

    # region Constructor
    @abstractmethod
    def __init__(self, dir_path: str):
        """
        Abstract Base Constructor for DataLoader class
        :param dir_path: String containing the full path to the data directory (can have multiple sub-folders)
        """
        self.path = dir_path  # Path of the directory containing the data (or sub-folders containing the data)
        self.filenames = None  # Placeholder for a list of filenames
        self.empty_error = None  # Placeholder for a file extension-specific loader error raised when the file contains headers only
    # endregion

    # region Methods
    def get_file_list(self, extension):
        """
        Create a list of all files within the path given by the directory (and its sub-folders) which have the input
        filename extension
        :param extension: Extension of the data files
        """
        self.filenames = glob.glob(os.path.join(self.path, '**', '*.' + extension), recursive=True)

    def load_data(self, read_kwargs: Optional[dict] = None):
        """
        Load the data from all relevant files in the path
        :param read_kwargs: Dictionary containing keyword arguments for the specific data loading method for the
        given file format used in the child class
        :return: A list of data arrays, each loaded by the specific child class
        """
        def print_if_empty():
            # Print to command line that current file was empty
            filename_end = re.split(r'\\', file_name)
            filename_end = filename_end[-2] + '\\' + filename_end[-1]
            print('file {} is empty and has been skipped'.format(filename_end))

        # Change immutable default to needed mutable default value
        if read_kwargs is None:
            read_kwargs = {}

        data = []
        for file_name in self.filenames:  # Loop over all data files
            try:
                # Try reading the file
                read_array, empty_flag = self.read_file(file_name, read_kwargs)
                if not empty_flag:
                    # If the file has data in it, append to the data list
                    data.append(read_array)
                else:
                    # Else skip file and print to command line
                    print_if_empty()
            except self.empty_error:  # Parsing error is raised due to file containing headers only
                # Skip file and print to command line
                print_if_empty()
            except Exception as error:
                raise error
        return data
    # endregion

    # region Abstract Methods
    @abstractmethod
    def to_numpy(self, data):
        """
        Convert data to list of numpy arrays
        :param data: List of objects of type determined by specific loading method
        :rtype: list
        """
        pass

    @abstractmethod
    def read_file(self, file_name, read_kwargs):
        """
        Read data file
        :param file_name: Full path of data file
        :param read_kwargs: Dictionary containing keyword arguments for the specific data loading method for the
        given file format used in the child class
        :rtype: tuple
        """
        pass
    # endregion
# endregion


# region csvLoader Concrete Subclass (DataLoader)
class csvLoader(DataLoader):
    """
    Concrete Subclass of DataLoader implementing loading data from csv files specifically
    """
    # region Constructor
    def __init__(self, dir_path: str):
        """
        Constructor method for csvLoader objects
        :param dir_path: String containing the full path to the data directory (can have multiple sub-folders)
        """
        super().__init__(dir_path)
        self.get_file_list('csv')  # Get list of csv files inside the path
        self.empty_error = pd.errors.ParserError  # Pandas parsing error raised when the file contains headers only
    # endregion

    # region Overriding Methods
    def to_numpy(self, data):
        """
        Convert list of pandas dataframes to list of numpy arrays
        :param data: List of pandas dataframes/series objects
        :return: List of numpy arrays
        """
        numpy_data = [d.to_numpy() for d in data]
        return numpy_data

    def read_file(self, file_name, read_kwargs):
        """
        Read csv data file
        :param file_name: Full path of csv file
        :param read_kwargs: Dictionary containing keyword arguments for pd.read_csv method
        :return: read_array - dataframe/series pandas object containing the data
                 empty_flag - bool specifying if the file was empty or not
        """
        read_array = pd.read_csv(file_name, **read_kwargs)
        empty_flag = False
        if read_array.empty:
            empty_flag = True
        return read_array, empty_flag
    # endregion
# endregion
