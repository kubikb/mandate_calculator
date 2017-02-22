import logging
import numpy as np
import helpers


# Custom exception
class MandateCalculatorException(Exception):
    pass


class MandateCalculator:

    NUM_POLITICAL_FORMATIONS = 4  # Number of political formations to handle
    NUM_SMDS = 106  # Number of SMDs
    VOTE_LIMIT = 0.05  # Ratio of minimal votes

    array_of_earlier_results = None  # Numpy array of results from the last election for every SMD
    num_smd_votes = None  # Numpy array of number of votes from the last election for every SMD
    factual_ratios = None  # Numpy array of the factual ratios from the last election
    predicted_ratios = None  # Numpy array of the user-predicted ratios (should be a column vector)
    votes_from_abroad = None  # Numpy array of votes coming from abroad for each political formation
    region_smd_array = None  # Numpy array of regional corrections for every SMD and political formation

    ratio_predicted_factual = None  # Numpy array of ratio of predicted/factual national vote values

    # Constructor
    def __init__(self,
                 array_of_earlier_results,
                 num_smd_votes,
                 factual_ratios,
                 predicted_ratios,
                 votes_from_abroad,
                 region_smd_array):

        logging.debug("Initializing MandateCalculator object is in progress...")

        self.__validate_smd_party_matrix(array_of_earlier_results)
        self.array_of_earlier_results = array_of_earlier_results
        logging.debug("The array of earlier results for every SMD is of size %s, %s." % array_of_earlier_results.shape)

        self.__validate_num_votes_smd(num_smd_votes)
        self.num_smd_votes = num_smd_votes
        logging.debug("The array of SMD vote counts from the last election successfully set!")

        self.__validate_vector(factual_ratios, human_readable_name="factual national ratios")
        self.factual_ratios = factual_ratios
        logging.debug("The array of factual national results is %s." % factual_ratios)

        self.__validate_vector(predicted_ratios, human_readable_name="predicted national ratios")
        self.predicted_ratios = predicted_ratios
        logging.debug("The array of predicted national results is %s." % predicted_ratios)

        self.__validate_vector(votes_from_abroad, human_readable_name="votes from abroad")
        self.votes_from_abroad = votes_from_abroad
        logging.debug("The array of votes from abroad is %s." % votes_from_abroad)

        self.__validate_smd_party_matrix(region_smd_array)
        self.region_smd_array = region_smd_array
        logging.debug("The array of SMD corrections for every SMD is of size %s, %s." % region_smd_array.shape)

        self.ratio_predicted_factual = self.__calculate_ratio_predicted_factual(predicted_ratios=predicted_ratios)

        logging.debug("MandateCalculator object was successfully initialized!")


    ##################### Public methods ###################
    # Calculate final list mandates
    def calculate_mandates(self, all_list_votes, smd_winners):
        dhondt_matrix, smallest_mandate_limit = helpers.dhondt_method(array_of_all_list_votes=all_list_votes,
                                                                      num_mandates=93)
        list_mandates = np.sum(dhondt_matrix >= smallest_mandate_limit, axis=0)
        sum_list_mandates = np.sum(list_mandates)
        if sum_list_mandates < 93:
            list_mandates /= sum_list_mandates
            list_mandates *= 93
        smd_mandates = np.sum(smd_winners, axis=0)
        return smd_mandates, list_mandates

    # Calculate SMD votes according to predicted national ratios
    def calculate_predicted_smd_ratios(self, ratio_predicted_factual = None, normalize_stepwise=True):
        if ratio_predicted_factual is None:
            ratio_predicted_factual = self.ratio_predicted_factual
        logging.debug("Calculating SMD votes according to predicted national ratios...")
        temp_array = np.tile(ratio_predicted_factual,
                             (self.NUM_SMDS, 1))
        if normalize_stepwise:
            normalized_smd_array = helpers.normalize_by_rows(self.array_of_earlier_results)
            temp_array *= normalized_smd_array
            normalized_predicted_smd_votes = helpers.normalize_by_rows(temp_array)
            normalized_predicted_smd_votes *= self.region_smd_array # Regional correction
            normalized_regional_votes = helpers.normalize_by_rows(normalized_predicted_smd_votes)
        else:
            temp_array *= self.array_of_earlier_results
            temp_array *= self.region_smd_array # Regional correction
            normalized_regional_votes = helpers.normalize_by_rows(temp_array)

        return normalized_regional_votes

    # Determine SMD winners
    def determine_smd_winners(self, smd_predicted_array):
        logging.debug("Determining SMD winners...")
        max_votes = np.max(smd_predicted_array,
                           axis=1,
                           keepdims=True)
        return np.equal(smd_predicted_array, max_votes)

    # Determine SMD second largest vote counts
    def determine_smd_second_largest_vote_ratios(self, smd_predicted_array):
        logging.debug("Determining SMD second largest vote ratios...")
        max_votes = np.max(smd_predicted_array,
                           axis=1,
                           keepdims=True)
        temp_array = np.not_equal(smd_predicted_array, max_votes)
        temp_array = temp_array * smd_predicted_array
        return np.max(temp_array, axis=1, keepdims=True)

    # Determine fractional votes without winner compensation
    def calculate_fractional_votes_without_winner_compensation(self, smd_predicted_array):
        logging.debug("Calculating fractional votes without winner compensation...")
        max_votes = np.max(smd_predicted_array,
                           axis=1,
                           keepdims=True)
        temp_array = np.not_equal(smd_predicted_array, max_votes)
        temp_array = temp_array * smd_predicted_array
        temp_array *= self.num_smd_votes
        return np.floor(temp_array)

    # Calculate fractional votes with winner compensation
    def calculate_fractional_votes_with_winner_compensation(self,
                                                            smd_predicted_array,
                                                            smd_winners,
                                                            smd_second_largest_votes,
                                                            fractional_votes_without_winner_comp):
        logging.debug("Calculating fractional votes with winner compensation...")
        max_votes = np.max(smd_predicted_array,
                           axis=1,
                           keepdims=True)
        max_votes = max_votes * smd_winners
        diff_largest_second_largest = max_votes - smd_second_largest_votes
        temp_array = diff_largest_second_largest * self.num_smd_votes
        temp_array *= smd_winners
        temp_array += fractional_votes_without_winner_comp
        return np.floor(temp_array)

    # Calculate all votes
    def calculate_all_list_votes(self,
                                 fractional_votes_with_winner_comp,
                                 predicted_ratios = None):
        logging.debug("Calculating all list votes...")
        if predicted_ratios is None:
            predicted_ratios = self.predicted_ratios

        # Calculate sum of fractional votes with winner compensation
        fractional_vote_sum = np.sum(fractional_votes_with_winner_comp, axis=0)

        # National list votes
        all_votes = np.sum(self.num_smd_votes, axis=0)
        all_votes = all_votes * predicted_ratios

        # Above the limit?
        above_limit = predicted_ratios >= self.VOTE_LIMIT
        all_list_votes = fractional_vote_sum + all_votes + self.votes_from_abroad
        return all_list_votes * above_limit

    # Helper function to calculate all mandates
    def calculate_all_mandates(self,
                               predicted_ratios = None,
                               ratio_predicted_factual = None):
        logging.debug("Calculating all mandates...")
        if ratio_predicted_factual is None:
            ratio_predicted_factual = self.ratio_predicted_factual
        if predicted_ratios is None:
            predicted_ratios = self.predicted_ratios
        predicted_smd_votes = self.calculate_predicted_smd_ratios(ratio_predicted_factual=ratio_predicted_factual)
        smd_winners = self.determine_smd_winners(smd_predicted_array=predicted_smd_votes)
        smd_second_largest_votes = self.determine_smd_second_largest_vote_ratios(
            smd_predicted_array=predicted_smd_votes)
        fractional_votes_without_winner_comp = self.calculate_fractional_votes_without_winner_compensation(
            smd_predicted_array=predicted_smd_votes)
        fractional_votes_with_winner_comp = self.calculate_fractional_votes_with_winner_compensation(
            smd_predicted_array=predicted_smd_votes,
            smd_winners=smd_winners,
            smd_second_largest_votes=smd_second_largest_votes,
            fractional_votes_without_winner_comp=fractional_votes_without_winner_comp)
        all_list_votes = self.calculate_all_list_votes(
            fractional_votes_with_winner_comp=fractional_votes_with_winner_comp,
            predicted_ratios=predicted_ratios)
        smd_mandates, list_mandates = self.calculate_mandates(all_list_votes=all_list_votes,
                                                              smd_winners=smd_winners)
        return smd_mandates, list_mandates

    # Function to calculate elasticities
    def calculate_elasticities(self, fixed_party_indicies, granularity=None, support_threshold=None):
        logging.debug("Calculating elasticities...")
        if not isinstance(fixed_party_indicies, list):
            raise MandateCalculatorException("Fixed party indices should be provided as a list!")
        if len(fixed_party_indicies) != 2:
            raise MandateCalculatorException("There should be two fixed parties!")

        party_indices = range(0, self.NUM_POLITICAL_FORMATIONS)
        non_fixed_parties = [item for item in party_indices if item not in fixed_party_indicies]
        available_support = 100 - float(np.sum(self.predicted_ratios[fixed_party_indicies])) * 100
        if granularity is None:
            granularity = 0.5
        results = []  # List to store results in
        for i in helpers.frange(0 + granularity, int(available_support), granularity):
            temp_array = self.predicted_ratios.copy()
            party_a_support = i / 100.0
            party_b_support = (available_support - i) / 100

            calculate = True
            if support_threshold is not None:
                if abs(party_a_support - party_b_support) > support_threshold:
                    calculate = False
            if calculate:
                temp_array[non_fixed_parties] = [party_a_support, party_b_support]
                smd_mandates, list_mandates = self.calculate_all_mandates(
                    ratio_predicted_factual=self.__calculate_ratio_predicted_factual(predicted_ratios=temp_array),
                    predicted_ratios=temp_array)
                smd_mandates += list_mandates
                relevant_mandates = smd_mandates[non_fixed_parties].tolist()
                result = [i, available_support - i] + relevant_mandates
                results.append(result)
        return results

    ##################### Private methods ###################
    # Calculate the ratio between predicted and factual national values
    def __calculate_ratio_predicted_factual(self, predicted_ratios):
        logging.debug("Calculating ratio between predicted and actual national ratios...")
        return predicted_ratios / self.factual_ratios

    # Function to validate array of earlier results
    def __validate_smd_party_matrix(self, array_of_earlier_results):

        logging.debug("Validating the array of results from the last election...")

        # Validate that array is in fact a numpy array
        if not isinstance(array_of_earlier_results, (np.ndarray, np.generic)):
            raise MandateCalculatorException("The provided array of results from the last election "
                                             "is not a numpy array!")
        else:
            logging.debug("The provided array of earlier results is in fact a numpy array!")

        # Validate size
        n_rows, n_cols = array_of_earlier_results.shape
        if n_rows != self.NUM_SMDS:
            raise MandateCalculatorException("The number of rows in the array of earlier results "
                                             "does not equal the number of SMDs!")

        elif n_cols != self.NUM_POLITICAL_FORMATIONS:
            raise MandateCalculatorException("The number of columns in the array of earlier results "
                                             "does not equal the number of political formations (%s)!"
                                             % self.NUM_POLITICAL_FORMATIONS)

        else:
            logging.debug("The provided array's size is correct!")

        return True

    # Function to validate array of earlier results
    def __validate_num_votes_smd(self, array_smd_vote_counts):

        logging.debug("Validating the array of SMD vote counts from the last election...")

        # Validate that array is in fact a numpy array
        if not isinstance(array_smd_vote_counts, (np.ndarray, np.generic)):
            raise MandateCalculatorException("The provided array of SMD vote counts from the last election "
                                             "is not a numpy array!")
        else:
            logging.debug("The provided array of SMD vote counts is in fact a numpy array!")

        # Validate size
        n_rows = array_smd_vote_counts.shape
        if len(n_rows) <= 1:
            raise MandateCalculatorException("The provided array of SMD vote counts is a vector!")

        elif n_rows[0] != self.NUM_SMDS:
            raise MandateCalculatorException("The number of rows in the array of SMD vote counts "
                                             "does not equal the number of SMDs!")

        else:
            logging.debug("The provided array's size is correct!")

        return True

    # Function to validate that the provided value is in fact a ratio
    def __validate_vector(self, array_to_validate, human_readable_name="votes"):

        # Validate that array is in fact a numpy array
        if not isinstance(array_to_validate, (np.ndarray, np.generic)):
            raise MandateCalculatorException("The provided array of %s is not a numpy array!"
                                             % human_readable_name)

        # Validate that it has the correct number of rows
        n_rows = array_to_validate.shape
        if len(n_rows) > 1:
            raise MandateCalculatorException("The provided array of %s is not a vector!"
                                             % human_readable_name)

        if n_rows[0] != self.NUM_POLITICAL_FORMATIONS:
            raise MandateCalculatorException("The number of rows in the array of %s "
                                             "does not equal the number of political formations (%s)!"
                                             % (human_readable_name,
                                                self.NUM_POLITICAL_FORMATIONS))

        return True