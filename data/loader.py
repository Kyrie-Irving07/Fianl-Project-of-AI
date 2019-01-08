import numpy as np


def data_process(indexes, times, attributes, values, results=None):
    label = []
    data = []
    template = np.transpose([indexes, np.zeros_like(indexes)])
    for i in range(len(times)):
        for j in range(len(attributes)):
            for k in range(len(values)):
                data_temp = template
                data_temp[i][1] = 1  # Time  label 1
                data_temp[j][1] = 2  # Attr  label 2
                data_temp[k][1] = 3  # Value label 3
                data.append(data_temp)
                if results:
                    label.append([i, j, k] in results)
    data = np.float32(data)
    label = np.float32(label)
    if results:
        return data, label
    else:
        return data
