import numpy as np
import copy


def data_process(indexes, times, attributes, values, results=None, max_input_length=200):
    label = []
    data = []
    mask = []
    template = np.transpose([indexes, np.zeros_like(indexes)])
    padding = np.zeros([max_input_length-len(indexes), 2])
    template = np.concatenate((template, padding), 0)
    for i in times:
        for j in attributes:
            for k in values:
                data_temp = copy.deepcopy(template)
                data_temp[i][1] = 1000  # Time  label 1000
                data_temp[j][1] = 2000  # Attr  label 2000
                data_temp[k][1] = 3000  # Value label 3000
                data.append(data_temp)
                mask.append([1, j, k])
                label.append(1 if [i, j, k] in results else -1)
    data = np.float32(data)
    label = np.float32(label)
    mask = np.int32(mask)
    # mask = list(range(len(indexes)))
    return data, label, mask
