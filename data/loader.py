import numpy as np


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
                data_temp = template
                data_temp[i][1] = 100  # Time  label 100
                data_temp[j][1] = 200  # Attr  label 200
                data_temp[k][1] = 300  # Value label 300
                data.append(data_temp)
                mask.append([i, j, k])
                label.append(1 if [i, j, k] in results else -1)
    mask = np.int32(mask)
    data = np.float32(data)
    label = np.float32(label)
    # mask = list(range(len(indexes)))
    return data, label, mask
