import unittest
import numpy as np
import numpy.testing as np_test
import mandate_calculator.helpers as helpers

class TestHelpers(unittest.TestCase):

    # Test normalize by rows func
    def test_normalize_by_rows(self):
        test_array = np.array([[0.0, 1.0],
                               [1.0, 3.0],
                               [1.0, 9.0]])
        normalized_array = helpers.normalize_by_rows(test_array)
        np_test.assert_equal(normalized_array,
                             np.array([[0.0, 1.0],
                                       [0.25, 0.75],
                                       [0.1, 0.9]]))

    # Test D'Hondt
    def test_dhondt_method(self):
        test_array = np.array([2230525, 2765331, 2512738, 457583])
        dhondt_matrix, smallest_mandate_limit = helpers.dhondt_method(array_of_all_list_votes=test_array,
                                                num_mandates=93)
        mandate_sums = np.sum(dhondt_matrix >= smallest_mandate_limit, axis=0)
        np_test.assert_equal(mandate_sums,
                             np.array([26, 33, 29, 5]))