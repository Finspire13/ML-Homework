import numpy as np
import pandas as pd


def load_data():
    attr_mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    class_mapping = {'draw': 17,
                     'zero': 0,
                     'one': 1,
                     'two': 2,
                     'three': 3,
                     'four': 4,
                     'five': 5,
                     'six': 6,
                     'seven': 7,
                     'eight': 8,
                     'nine': 9,
                     'ten': 10,
                     'eleven': 11,
                     'twelve': 12,
                     'thirteen': 13,
                     'fourteen': 14,
                     'fifteen': 15,
                     'sixteen': 16}

    data_file = './Dataset/krkopt.data'
    data = pd.read_csv(data_file, header=None)
    data = np.array(data)

    for key, val in attr_mapping.items():
        data[data == key] = val
    for key, val in class_mapping.items():
        data[data == key] = val

    data = data.astype('int')

    return {'data': data[:, :-1],
            'target': data[:, -1]}
