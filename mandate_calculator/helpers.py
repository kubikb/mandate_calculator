import numpy as np
import logging

# Function to normalize array by rows
def normalize_by_rows(array):
    row_sums = np.sum(array,
                      axis=1,
                      keepdims=True)
    array /= row_sums
    return array

# D'Hondt method
def dhondt_method(array_of_all_list_votes,
                  num_mandates=93):
    logging.debug("Calculating D'Hondt mandates for array %s..." % array_of_all_list_votes)
    array_of_divisors = np.arange(1, 61)
    array_of_divisors = np.expand_dims(array_of_divisors, axis=1)
    dhondt_matrix = np.divide(np.tile(array_of_all_list_votes,(60, 1)),
                              array_of_divisors)
    temp_array = np.sort(dhondt_matrix.ravel())
    temp_array = temp_array[np.nonzero(temp_array)]
    if len(temp_array) < num_mandates:
        smallest_mandate_limit = temp_array[-1]
    else:
        smallest_mandate_limit = temp_array[-num_mandates]
    logging.debug("Smallest limit for D'Hondt mandates was determined to be %s." % smallest_mandate_limit)
    return dhondt_matrix, smallest_mandate_limit

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump
