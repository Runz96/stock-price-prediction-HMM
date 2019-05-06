import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from docopt import docopt


def extract_features(data):
    open_price = np.array(data['Open'])
    close_price = np.array(data['Close'])
    high_price = np.array(data['High'])
    low_price = np.array(data['Low'])

    # Compute the fraction change in close, high and low prices
    # which would be used a feature
    frac_change = (close_price - open_price) / open_price
    frac_high = (high_price - open_price) / open_price
    frac_low = (open_price - low_price) / open_price

    return np.column_stack((frac_change, frac_high, frac_low))


def compute_all_possible_outcomes(num_steps_frac_change,num_steps_frac_high, num_steps_frac_low):
    frac_change_range = np.linspace(-0.1, 0.1, num_steps_frac_change)
    frac_high_range = np.linspace(0, 0.1, num_steps_frac_high)
    frac_low_range = np.linspace(0, 0.1, num_steps_frac_low)
    possible_outcomes = np.array(list(itertools.product(frac_change_range, frac_high_range, frac_low_range)))

    return possible_outcomes


if __name__ == '__main__':
    test_size = 0.33
    num_states = 4
    num_latency_days = 10
    n_latency_days = 10
    num_steps_frac_change = 50
    num_steps_frac_high = 10
    num_steps_frac_low = 10
    data = pd.read_csv('C:/Users/Runz/Desktop/BMLProject/Apple.csv')
    train_set, test_set = train_test_split(data, test_size=test_size, shuffle=False)

    # extract features
    feature_vector = extract_features(train_set)

    hmm = GaussianHMM(n_components=num_states)
    hmm.fit(feature_vector)

    possible_outcomes = compute_all_possible_outcomes(num_steps_frac_change, num_steps_frac_high, num_steps_frac_low)



