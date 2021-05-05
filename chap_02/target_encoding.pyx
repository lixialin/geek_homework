# distutils: language=c++
import numpy as np
cimport numpy as np


cpdef target_mean_v3(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[int] y = np.asfortranarray(data[y_name], dtype=np.int)
    cdef np.ndarray[int] x = np.asfortranarray(data[x_name], dtype=np.int)

    cdef dict value_dict = dict()
    cdef dict count_dict = dict()

    cdef long i
    for i in range(nrow):
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1

    cdef long j
    for j in range(nrow):
        result[j] = (value_dict[x[j]] - y[j])/(count_dict[x[j]]-1)

    return result

