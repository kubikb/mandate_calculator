# Mandate Calculator
[![Build Status](https://travis-ci.org/kubikb/mandate_calculator.svg?branch=master)](https://travis-ci.org/kubikb/mandate_calculator)

This repository contains my Python implementation for the [mandate calculator](https://mandatumkalkulator.herokuapp.com/) model designed by Dániel Róna. For the detailed description of the model, please read the [methodology page](https://mandatumkalkulator.herokuapp.com/methodology).

## Prerequisites
1. Python 2.7 and Pip should be installed. On Windows, install Anaconda for Python 2.7 (https://www.continuum.io/downloads) and you'll be ready to roll.
2. Install the required Python libraries (don't do this if you use Anaconda): `pip install -r requirements.txt`

## Running it

### Mandate calculation
The `MandateCalculator` class contains all the methods needed to do a calculation. Initialize a `MandateCalculator` object by giving it the following variables:

- `array_of_earlier_results` - This should be a numpy array of size 106*4 (since there are 106 SMDs and 4 political formations in the model). The ratios in this matrix represent the factual vote ratios from the latest election.
- `num_smd_votes` - Numpy matrix of number of votes from the last election for every SMD. Shape should be (106, 1)
- `factual_ratios` - A numpy array of shape (4,) that represents the national vote ratios obtained by each political formation
- `predicted_ratios` - User-predicted national vote ratios for each political formation. Also a numpy array of shape (4,)
- `votes_from_abroad` - Numpy array of votes coming from abroad for each political formation. Also a numpy array of shape (4,)
- `region_smd_array` - Numpy array of regional corrections for every SMD and political formation. Shape should be (106, 1)

An example on how to run it:
```
from mandate_calculator.model import MandateCalculator, MandateCalculatorException
import numpy as np

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
                                            
factual_ratios = np.array([0.2669, 0.4355, 0.2029, 0.0547])
predicted_ratios = np.array([0.26, 0.34, 0.33, 0.05])
votes_from_abroad = np.array([1495, 122638, 2926, 574])

model = MandateCalculator(array_of_earlier_results=array_of_earlier_results,
                                      num_smd_votes=array_of_smd_vote_counts,
                                      factual_ratios=factual_ratios,
                                      predicted_ratios=predicted_ratios,
                                      votes_from_abroad=votes_from_abroad,
                                      region_smd_array=array_of_regional_corrections)
smd_mandates, list_mandates = model.calculate_all_mandates()
```

### Tests
Execute `nosetests` in the cloned directory. This will run all the unittests.

### Performance metrics
You can get some performance metrics by running `python test_speed.py`
```
Executing model 10 times took 0.0140001773834 seconds (714.276664226 model calculations per second)!
Executing model 100 times took 0.152000188828 seconds (657.89391955 model calculations per second)!
Executing model 1000 times took 1.49800014496 seconds (667.556677725 model calculations per second)!
Executing model 10000 times took 14.8910000324 seconds (671.546570292 model calculations per second)!
```
