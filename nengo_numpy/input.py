
import numpy as np

# class Input(object):
#     def __init__(self, dimensions, filter=None):
#         self.set([0] * dimensions)
#         self.filter = filter
#     def update(self, dt):
#         if self.filter is not None: 
#             self.state = self.filter.update(self.raw, dt)
#         else:
#             self.state = self.raw    
#     def set(self, value):
#         self.raw = np.array(value)
#     def get(self):
#         return self.state
#     def reset(self):
#         self.set([0] * len(self.state))

def clean_value(value):
    # return np.asarray(value).astype('float32').flatten()
    return np.asarray(value).flatten()

class Input(object):

    def __init__(self, name, value):

        self.name = name
        self.change_time = None

        # if value parameter is a python function
        # if callable(value):
            # self.origin['X'] = origin.Origin(func=value)

        # if value is dict of time:value pairs
        if isinstance(value, dict):
            self.keys = sorted(value.keys())
            self.dict = \
                dict((k, clean_value(value[k])) for k in self.keys)
            self.change_time = self.keys[0]
            self._value = self.dict[self.change_time].copy()
        else:
            self._value = clean_value(value).copy()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, x):
        self._value[:] = x

    @property
    def output(self):
        return self._value

    def reset(self):
        if self.change_time is not None:
            self.change_time = self.keys[0]
            self.value = self.dict[self.change_time]

    def reset_input(self):
        pass

    def run(self, t_end):
        if self.change_time is not None and t_end > self.change_time:
            self.value = self.dict[self.change_time]

            ### set change_time to next time after t_end, None if DNE
            self.change_time = next((t for t in self.keys if t > t_end), None)
