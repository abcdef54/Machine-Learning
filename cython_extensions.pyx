cpdef double score_regressor(double[:] y_test, double[:] predictions):
    cdef double ss_res = 0.0
    cdef double ss_tot = 0.0
    cdef double mean = cmean(y_test)
    cdef int i
    cdef int length = len(predictions)
    cdef int lengthy = len(y_test)

    if lengthy != length:
        raise ValueError("y_test and predictions must have the same length")

    for i in range(length):
        ss_res += (y_test[i] - predictions[i]) ** 2
        ss_tot += (y_test[i] - mean) ** 2

    if ss_tot == 0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)

cpdef double cmean(double[:] array):
    cdef int length = len(array)
    cdef int i = 0
    cdef double total = 0
    for i in range(length):
        total += array[i]
    return total / length


