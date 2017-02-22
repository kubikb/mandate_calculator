import unittest
from mandate_calculator.model import MandateCalculator, MandateCalculatorException
import pandas as pd
import os
import numpy as np
import numpy.testing as np_test

current_dir = os.path.split(os.path.abspath(__file__))[0]
array_of_earlier_results = pd.read_csv(os.path.join(current_dir, "files/results_2014.csv"),
                                       sep=";",
                                       encoding="UTF-8",
                                       index_col=[0,1],
                                       decimal=",").as_matrix()
array_of_smd_vote_counts = pd.read_csv(os.path.join(current_dir, "files/smd_counts_2014.csv"),
                                       sep=";",
                                       encoding="UTF-8",
                                       index_col=0,
                                       decimal=",").as_matrix()
array_of_regional_corrections = pd.read_csv("mandate_calculator/tests/files/smd_region.csv",
                                            sep=";",
                                            encoding="UTF-8",
                                            index_col=0,
                                            decimal=",").as_matrix()

factual_ratios = np.array([0.2669, 0.4355, 0.2029, 0.0547])
predicted_ratios = np.array([0.26, 0.34, 0.33, 0.05])
votes_from_abroad = np.array([1495, 122638, 2926, 574])

class TestMandateCalculator(unittest.TestCase):

    # Test that list as array of earlier results raises exception
    def test_result_array_type(self):
        with self.assertRaisesRegexp(MandateCalculatorException,
                                     "numpy array"):
            MandateCalculator(array_of_earlier_results=array_of_earlier_results.tolist(),
                              num_smd_votes=array_of_smd_vote_counts,
                              factual_ratios=factual_ratios,
                              predicted_ratios=predicted_ratios,
                              votes_from_abroad=votes_from_abroad,
                              region_smd_array=array_of_regional_corrections)

    # Test that providing incorrect number of political formations raises an exception
    def test_result_array_pol_formations(self):
        with self.assertRaisesRegexp(MandateCalculatorException,
                                     "political formations"):
            test_array = array_of_earlier_results.copy()
            test_array = np.delete(test_array, 1, 1)
            MandateCalculator(array_of_earlier_results=test_array,
                              num_smd_votes=array_of_smd_vote_counts,
                              factual_ratios=factual_ratios,
                              predicted_ratios=predicted_ratios,
                              votes_from_abroad=votes_from_abroad,
                              region_smd_array=array_of_regional_corrections)

    # Test that providing incorrect number of SMDs raises an exception
    def test_result_array_smds(self):
        with self.assertRaisesRegexp(MandateCalculatorException,
                                     "SMDs"):
            test_array = array_of_earlier_results.copy()
            test_array = np.delete(test_array, 1, 0)
            MandateCalculator(array_of_earlier_results=test_array,
                              num_smd_votes=array_of_smd_vote_counts,
                              factual_ratios=factual_ratios,
                              predicted_ratios=predicted_ratios,
                              votes_from_abroad=votes_from_abroad,
                              region_smd_array=array_of_regional_corrections)

    # Test that SMD count as a list raises an exception
    def test_smd_count_array_type(self):
        with self.assertRaisesRegexp(MandateCalculatorException,
                                     "SMD vote"):
            MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                              num_smd_votes=array_of_smd_vote_counts.tolist(),
                              factual_ratios=factual_ratios,
                              predicted_ratios=predicted_ratios,
                              votes_from_abroad=votes_from_abroad,
                              region_smd_array=array_of_regional_corrections)

    # Test that SMD count as a vector raises an exception
    def test_smd_count_array_vector(self):
        with self.assertRaisesRegexp(MandateCalculatorException,
                                     "vector"):
            MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                              num_smd_votes=array_of_smd_vote_counts.flatten(),
                              factual_ratios=factual_ratios,
                              predicted_ratios=predicted_ratios,
                              votes_from_abroad=votes_from_abroad,
                              region_smd_array=array_of_regional_corrections)

    # Test that list as array of earlier results raises exception
    def test_smd_count_array_smds(self):
        with self.assertRaisesRegexp(MandateCalculatorException,
                                     "rows"):
            test_array = array_of_smd_vote_counts.copy()
            test_array = np.delete(test_array, 1, 0)
            MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                              num_smd_votes=test_array,
                              factual_ratios=factual_ratios,
                              predicted_ratios=predicted_ratios,
                              votes_from_abroad=votes_from_abroad,
                              region_smd_array=array_of_regional_corrections)

    # Test that list as array of ratios raises exception
    def test_ratio_array_type(self):
        with self.assertRaisesRegexp(MandateCalculatorException,
                                     "numpy array"):
            MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                              num_smd_votes=array_of_smd_vote_counts,
                              factual_ratios=factual_ratios.tolist(),
                              predicted_ratios=predicted_ratios,
                              votes_from_abroad=votes_from_abroad,
                              region_smd_array=array_of_regional_corrections)

    # Test that matrix as array ratios raises exception
    def test_ratio_array_column(self):
        with self.assertRaisesRegexp(MandateCalculatorException,
                                     "vector"):
            MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                              num_smd_votes=array_of_smd_vote_counts,
                              factual_ratios=np.array([[0.2669, 0.4355, 0.2029, 0.0547]]).transpose(),
                              predicted_ratios=predicted_ratios,
                              votes_from_abroad=votes_from_abroad,
                              region_smd_array=array_of_regional_corrections)

    # Test that incorrect size of ratios array raises exception
    def test_ratio_array_pol_formations(self):
        with self.assertRaisesRegexp(MandateCalculatorException,
                                     "political formations"):
            MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                              num_smd_votes=array_of_smd_vote_counts,
                              factual_ratios=np.delete(factual_ratios, 1, 0),
                              predicted_ratios=predicted_ratios,
                              votes_from_abroad=votes_from_abroad,
                              region_smd_array=array_of_regional_corrections)

    # Test calculate SMD votes according to predicted national ratios
    def test_calculate_predicted_smd_ratios(self):
        model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                  num_smd_votes=array_of_smd_vote_counts,
                                  factual_ratios=factual_ratios,
                                  predicted_ratios=predicted_ratios,
                                  votes_from_abroad=votes_from_abroad,
                                  region_smd_array=array_of_regional_corrections)
        predicted_smd_votes = model.calculate_predicted_smd_ratios()
        array_of_correct_pred_smd_votes = pd.read_csv(os.path.join(current_dir, "files/predicted_smd_votes.csv"),
                                                      sep=";",
                                                      encoding="UTF-8",
                                                      decimal=",",
                                                      header=None).as_matrix()
        np_test.assert_almost_equal(predicted_smd_votes,
                                    array_of_correct_pred_smd_votes,
                                    decimal=3)

    # Compare SMD votes between stepwise normalization and only a single normalization in the end
    def test_predicted_smd_ratios_normalization(self):
        model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                  num_smd_votes=array_of_smd_vote_counts,
                                  factual_ratios=factual_ratios,
                                  predicted_ratios=predicted_ratios,
                                  votes_from_abroad=votes_from_abroad,
                                  region_smd_array=array_of_regional_corrections)
        predicted_smd_votes_stepwise = model.calculate_predicted_smd_ratios()
        predicted_smd_votes = model.calculate_predicted_smd_ratios(normalize_stepwise=False)
        np_test.assert_almost_equal(predicted_smd_votes,
                                    predicted_smd_votes_stepwise,
                                    decimal=3)

    # Test SMD winner prediction
    def test_determine_smd_winners(self):
        model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                  num_smd_votes=array_of_smd_vote_counts,
                                  factual_ratios=factual_ratios,
                                  predicted_ratios=predicted_ratios,
                                  votes_from_abroad=votes_from_abroad,
                                  region_smd_array=array_of_regional_corrections)
        predicted_smd_votes = model.calculate_predicted_smd_ratios()
        predicted_winners = model.determine_smd_winners(smd_predicted_array=predicted_smd_votes)
        self.assertEqual(np.max(predicted_winners), 1)
        array_of_correct_smd_winners = pd.read_csv(os.path.join(current_dir, "files/smd_winners.csv"),
                                                   sep=";",
                                                   encoding="UTF-8",
                                                   decimal=",",
                                                   header=None).as_matrix()
        np_test.assert_almost_equal(predicted_winners,
                                    array_of_correct_smd_winners,
                                    decimal=3)

    # Test second largest vote calculator
    def test_determine_smd_second_largest_vote_ratios(self):
        model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                  num_smd_votes=array_of_smd_vote_counts,
                                  factual_ratios=factual_ratios,
                                  predicted_ratios=predicted_ratios,
                                  votes_from_abroad=votes_from_abroad,
                                  region_smd_array=array_of_regional_corrections)
        predicted_smd_votes = model.calculate_predicted_smd_ratios()
        second_largest_votes = model.determine_smd_second_largest_vote_ratios(smd_predicted_array=predicted_smd_votes)
        array_of_correct_smd_second_largest_vote_ratios = pd.read_csv(os.path.join(current_dir, "files/smd_second_largest_vote_ratios.csv"),
                                                                      sep=";",
                                                                      encoding="UTF-8",
                                                                      decimal=",",
                                                                      header=None).as_matrix()
        np_test.assert_almost_equal(second_largest_votes,
                                    array_of_correct_smd_second_largest_vote_ratios,
                                    decimal=4)

    # Test fractional votes without winner compensation
    def test_calculate_fractional_votes_without_winner_compensation(self):
        model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                  num_smd_votes=array_of_smd_vote_counts,
                                  factual_ratios=factual_ratios,
                                  predicted_ratios=predicted_ratios,
                                  votes_from_abroad=votes_from_abroad,
                                  region_smd_array=array_of_regional_corrections)
        predicted_smd_votes = model.calculate_predicted_smd_ratios()
        fractional_votes_without_winner_comp = model.calculate_fractional_votes_without_winner_compensation \
            (smd_predicted_array=predicted_smd_votes)
        array_of_correct_fractional_votes_without_winner_compensation = pd.read_csv(
            os.path.join(current_dir, "files/fractional_votes_without_winner_compensation.csv"),
            sep=";",
            encoding="UTF-8",
            decimal=",",
            header=None).as_matrix()
        np_test.assert_almost_equal(fractional_votes_without_winner_comp,
                                    array_of_correct_fractional_votes_without_winner_compensation,
                                    decimal=0)

    # Test fractional votes with winner compensation
    def test_calculate_fractional_votes_with_winner_compensation(self):
        model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                  num_smd_votes=array_of_smd_vote_counts,
                                  factual_ratios=factual_ratios,
                                  predicted_ratios=predicted_ratios,
                                  votes_from_abroad=votes_from_abroad,
                                  region_smd_array=array_of_regional_corrections)
        predicted_smd_votes = model.calculate_predicted_smd_ratios()
        smd_winners = model.determine_smd_winners(smd_predicted_array=predicted_smd_votes)
        smd_second_largest_votes = model.determine_smd_second_largest_vote_ratios(smd_predicted_array=predicted_smd_votes)
        fractional_votes_without_winner_comp = model.calculate_fractional_votes_without_winner_compensation(smd_predicted_array=predicted_smd_votes)
        fractional_votes_with_winner_comp = model.calculate_fractional_votes_with_winner_compensation(
            smd_predicted_array=predicted_smd_votes,
            smd_winners=smd_winners,
            smd_second_largest_votes=smd_second_largest_votes,
            fractional_votes_without_winner_comp=fractional_votes_without_winner_comp)
        array_of_correct_fractional_votes_with_winner_compensation = pd.read_csv(
            os.path.join(current_dir, "files/fractional_votes_with_winner_compensation.csv"),
            sep=";",
            encoding="UTF-8",
            decimal=",",
            header=None).as_matrix()
        np_test.assert_almost_equal(fractional_votes_with_winner_comp,
                                    array_of_correct_fractional_votes_with_winner_compensation,
                                    decimal=0)

    # Test calculating all list votes
    def test_calculate_all_list_votes(self):
        model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                  num_smd_votes=array_of_smd_vote_counts,
                                  factual_ratios=factual_ratios,
                                  predicted_ratios=predicted_ratios,
                                  votes_from_abroad=votes_from_abroad,
                                  region_smd_array=array_of_regional_corrections)
        predicted_smd_votes = model.calculate_predicted_smd_ratios()
        smd_winners = model.determine_smd_winners(smd_predicted_array=predicted_smd_votes)
        smd_second_largest_votes = model.determine_smd_second_largest_vote_ratios(smd_predicted_array=predicted_smd_votes)
        fractional_votes_without_winner_comp = model.calculate_fractional_votes_without_winner_compensation(
            smd_predicted_array=predicted_smd_votes)
        fractional_votes_with_winner_comp = model.calculate_fractional_votes_with_winner_compensation(
            smd_predicted_array=predicted_smd_votes,
            smd_winners=smd_winners,
            smd_second_largest_votes=smd_second_largest_votes,
            fractional_votes_without_winner_comp=fractional_votes_without_winner_comp)
        all_list_votes = model.calculate_all_list_votes(fractional_votes_with_winner_comp=fractional_votes_with_winner_comp)
        np_test.assert_almost_equal(all_list_votes,
                                    np.array([2224688, 2751296, 2515307, 456522]),
                                    decimal=0)

    # Test calculating mandates
    def test_calculate_mandates(self):
        model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                  num_smd_votes=array_of_smd_vote_counts,
                                  factual_ratios=factual_ratios,
                                  predicted_ratios=predicted_ratios,
                                  votes_from_abroad=votes_from_abroad,
                                  region_smd_array=array_of_regional_corrections)
        predicted_smd_votes = model.calculate_predicted_smd_ratios()
        smd_winners = model.determine_smd_winners(smd_predicted_array=predicted_smd_votes)
        smd_second_largest_votes = model.determine_smd_second_largest_vote_ratios(
            smd_predicted_array=predicted_smd_votes)
        fractional_votes_without_winner_comp = model.calculate_fractional_votes_without_winner_compensation(
            smd_predicted_array=predicted_smd_votes)
        fractional_votes_with_winner_comp = model.calculate_fractional_votes_with_winner_compensation(
            smd_predicted_array=predicted_smd_votes,
            smd_winners=smd_winners,
            smd_second_largest_votes=smd_second_largest_votes,
            fractional_votes_without_winner_comp=fractional_votes_without_winner_comp)
        all_list_votes = model.calculate_all_list_votes(
            fractional_votes_with_winner_comp=fractional_votes_with_winner_comp)
        smd_mandates, list_mandates = model.calculate_mandates(all_list_votes=all_list_votes,
                                                               smd_winners=smd_winners)
        np_test.assert_equal(smd_mandates + list_mandates,
                             np.array([45, 75, 74, 5]))

    # Test that list mandates is correct in case of a single/dominant party
    def test_list_mandates_dominant_party(self):
        model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                  num_smd_votes=array_of_smd_vote_counts,
                                  factual_ratios=factual_ratios,
                                  predicted_ratios=np.array([0.88, 0.04, 0.04, 0.04]),
                                  votes_from_abroad=votes_from_abroad,
                                  region_smd_array=array_of_regional_corrections)
        predicted_smd_votes = model.calculate_predicted_smd_ratios()
        smd_winners = model.determine_smd_winners(smd_predicted_array=predicted_smd_votes)
        smd_second_largest_votes = model.determine_smd_second_largest_vote_ratios(
            smd_predicted_array=predicted_smd_votes)
        fractional_votes_without_winner_comp = model.calculate_fractional_votes_without_winner_compensation(
            smd_predicted_array=predicted_smd_votes)
        fractional_votes_with_winner_comp = model.calculate_fractional_votes_with_winner_compensation(
            smd_predicted_array=predicted_smd_votes,
            smd_winners=smd_winners,
            smd_second_largest_votes=smd_second_largest_votes,
            fractional_votes_without_winner_comp=fractional_votes_without_winner_comp)
        all_list_votes = model.calculate_all_list_votes(
            fractional_votes_with_winner_comp=fractional_votes_with_winner_comp)
        smd_mandates, list_mandates = model.calculate_mandates(all_list_votes=all_list_votes,
                                                               smd_winners=smd_winners)
        np_test.assert_equal(list_mandates,
                             np.array([93,0,0,0]))

    # Test that calculate elasticities accepts floats as granularity step
    def test_simulation_granularity(self):
        model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                  num_smd_votes=array_of_smd_vote_counts,
                                  factual_ratios=factual_ratios,
                                  predicted_ratios=np.array([0.88, 0.04, 0.04, 0.04]),
                                  votes_from_abroad=votes_from_abroad,
                                  region_smd_array=array_of_regional_corrections)
        try:
            model.calculate_elasticities(fixed_party_indicies=[0,1],
                                         granularity=0.5)
        except Exception, e:
            self.fail(e)

