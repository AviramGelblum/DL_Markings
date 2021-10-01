from typing import Optional
from enum import Enum


# region IProcessable Interface
class IProcessable:
    """
    Interface to be implemented by classes whose instances are to be processed in a Pipeline
    """
    def verify_type(self, instance_types: Optional[list[Enum]]):
        """
        Abstract method whose goal is to verify that the current object type is in a list of accepted instance-types.
        This method is called during processing of a Pipeline object. Some methods in the pipe should process only
        objects of a certain type.
        :param instance_types: List of relevant enum types or None
        :rtype: bool
        """
        pass
# endregion


# region Pipeline Class
class Pipeline:
    """
    Class implementing a sequential processing pipeline for a list of commands specifying methods.
    """
    def __init__(self, command_list: list):
        """
        Constructor method for the Pipeline class
        :param command_list: list of methods and their arguments, to be processed using input instances of the
        appropriate classes.
        """
        # List of commands stated by 4-tuples of the format (method, list of method arguments,
        # dictionary of keyword arguments, list of enum types selecting object instances to perform method on(None if
        # irrelevant))
        self.command_list = command_list
        self._validate_command_list()  # Validate the command list's method's number of arguments

    def _validate_command_list(self):
        """
        Verify that the number of arguments provided for each method is indeed the number of arguments that method
        expects to receive.
        """
        for command_entry in self.command_list:  # Command list format: (method, args, kwargs, [enum types])
            method = command_entry[0]

            number_of_kwargs = method.__defaults__  # Number of kwargs the method can accept
            # Calculate number of mandatory arguments the method expects
            if number_of_kwargs is None:
                number_of_kwargs = []
            min_num_args_in_method = method.__code__.co_argcount - len(number_of_kwargs)-1

            num_args_in_method_command = command_entry[1].__len__()  # Arguments provided in command list
            if min_num_args_in_method > num_args_in_method_command:
                # Raise error if not enough arguments are provided in the command entry
                num_missing_arguments = min_num_args_in_method-num_args_in_method_command
                raise TypeError('missing ' + str(num_missing_arguments) + ' arguments for method ' +
                                method.__name__ + ' in pipe')

    def process(self, instances):
        """
        Process the methods specified in the command list sequentially, over the appropriate instances.
        :param instances: Instances for which the appropriate methods provided in the command_list would be executed
        """
        for m in self.command_list:
            print('Running ' + m[0].__name__)
            method_class_name = m[0].__qualname__.split('.')[-2]  # Get the name of the class wherein the method is defined
            for instance in instances:
                # Get names of the current instance's class and all the classes it inherits from
                instance_class_names = {base_class.__name__ for base_class in instance.__class__.__bases__}
                instance_class_names.add(instance.__class__.__name__)

                # Check if the current method is defined in a class which the current instance is an instance of. i.e.
                # check whether the current instance can execute said method.
                if method_class_name in instance_class_names:
                    # If so, verify that the current instance's enum type is in the list of accepted types.
                    if instance.verify_type(m[3]):
                        # If so, execute the method for the current instance.
                        m[0](instance, *m[1], **m[2])

                        # print('\nProcessing history:')
                        # print(instance.history)
# endregion
