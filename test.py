import pandas as pd
import math
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    test_size = 0.33
    num_states = 4
    num_latency_days = 10
    n_latency_days = 10
    num_steps_frac_change = 50
    num_steps_frac_high = 10
    num_steps_frac_low = 10
    Bdata = pd.read_csv('C:/Users/Runz/Desktop/BMLProject/Apple.csv')
    #Adata = Bdata[0:809]
    train_set, test_set = train_test_split(Bdata, test_size=test_size, shuffle=False)
    print(len(Bdata))
    print(len(train_set))
    print(len(test_set))
    num = 8
    for i in range(num):
        print(i)



