import numpy as np


def data_process(indexes, times, attributes, values, results=None, max_input_length=200):
    label = []
    data = []
    template = np.transpose([indexes, np.zeros_like(indexes)])
    padding = np.zeros([max_input_length-len(indexes), 2])
    template = np.concatenate((template, padding), 0)
    for i in times:
        for j in attributes:
            for k in values:
                data_temp = template
                data_temp[i][1] = 1  # Time  label 1
                data_temp[j][1] = 2  # Attr  label 2
                data_temp[k][1] = 3  # Value label 3
                data.append(data_temp)
                label.append([i, j, k] in results)
    data = np.float32(data)
    label = np.float32(label)
    mask = list(range(len(indexes)))
    return data, label, mask
