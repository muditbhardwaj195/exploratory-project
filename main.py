import pandas as pd
import numpy
import os
import math
import csv
from load_datasets import load_datasets
from load_final_mrl import load_final_mrl
from create_datasets import make_test_cases
from gaussian_process_classifier import calculate_MRD
from time_series_prediction import check
from time_series_prediction import calculate_accuracy

# make sensor dataset and test case file on which our code will be tested
# make_test_cases()

#lets calculate mrl for each of the five sensors 1,10,19,28,37
# calculate_MRD()

#now lets classify time series from our test file
check()

#for calculating accuracy
calculate_accuracy()