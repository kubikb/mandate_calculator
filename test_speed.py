from mandate_calculator.model import MandateCalculator, MandateCalculatorException
import pandas as pd
import os, sys
import numpy as np
import time

current_dir = os.path.split(os.path.abspath(__file__))[0]
array_of_earlier_results = pd.read_csv("mandate_calculator/tests/files/results_2014.csv",
                                       sep=";",
                                       encoding="UTF-8",
                                       index_col=[0,1],
                                       decimal=",").as_matrix()
array_of_smd_vote_counts = pd.read_csv("mandate_calculator/tests/files/smd_counts_2014.csv",
                                       sep=";",
                                       encoding="UTF-8",
                                       index_col=0,
                                       decimal=",").as_matrix()
array_of_regional_corrections = pd.read_csv("mandate_calculator/tests/files/smd_region.csv",
                                            sep=";",
                                            encoding="UTF-8",
                                            index_col=0,
                                            decimal=",").as_matrix()
votes_from_abroad = np.array([1495, 122638, 2926, 574])

num_times = [10, 100, 1000, 10000]

if __name__ == "__main__":
    start = time.time()

    for t in num_times:
        for i in range(0, t):
            # Random arrays
            factual_ratios = np.random.uniform(size=4)
            factual_ratios = factual_ratios / np.sum(factual_ratios)
            predicted_ratios = np.random.uniform(size=4)
            predicted_ratios = predicted_ratios / np.sum(predicted_ratios)

            # Calculate model
            model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                      num_smd_votes=array_of_smd_vote_counts,
                                      factual_ratios=factual_ratios,
                                      predicted_ratios=predicted_ratios,
                                      votes_from_abroad=votes_from_abroad,
                                      region_smd_array=array_of_regional_corrections)
            smd_mandates, list_mandates = model.calculate_all_mandates()

        end = time.time()
        print "Executing model %s times took %s seconds (%s model calculations per second)!" %(t,
                                                                                               end-start,
                                                                                               t / (end-start))
