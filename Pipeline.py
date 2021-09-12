class Iprocessable:
    def verify_type(self, instance_types):
        pass


class Pipeline:
    def __init__(self, pipe_list: list):
        self.pipe = pipe_list
        self.__validate()

    def __validate(self):
        for pipe_entry in self.pipe:
            method = pipe_entry[0]
            kwargs = method.__defaults__
            if kwargs is None:
                min_num_args_in_method = method.__code__.co_argcount - 1
            else:
                min_num_args_in_method = method.__code__.co_argcount - len(method.__defaults__)-1

            num_args_in_method_pipe = pipe_entry[1].__len__()
            if min_num_args_in_method > num_args_in_method_pipe:
                num_missing_arguments = min_num_args_in_method-num_args_in_method_pipe
                raise TypeError('missing ' + str(num_missing_arguments) + ' arguments for method ' + \
                                method.__name__ + ' in pipe')

    def process(self, instances):
        for mm in enumerate(self.pipe):
            m = mm[1]
            print('Running ' + m[0].__name__)
            class_name = m[0].__qualname__.split('.')[-2]
            for instance in instances:
                if instance.__class__.__name__ == class_name:
                    if instance.verify_type(m[3]):
                        m[0](instance, *m[1], **m[2])
                        print(instance.history)

