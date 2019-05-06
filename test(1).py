import numpy as np
#from cvxopt import matrix, solvers
#import tensorflow as tf
from numpy import random

import cmath
#low dimentional change point
#
#this is the code for one variation
def changepoint2(q):
    n = len(q)
    XXX = []
    XXXX =[]
    for i in range(n):
        if i ==0:
            Wk = q[0]
        else:
            Wk = np.sum(q[0:i])
        sum2 = np.sum(q)
        summ = Wk/cmath.sqrt(n)-(i+1)*sum2/pow(n,1.5)
        Sum = abs(summ)
        K = pow(1000*(i+1)*(1-(i+1)/n)/n,0.5)
        KK = Sum/K
        XXX.append(KK)
    del XXX[-1]
    ##print(XXX)
    
    for i in range(200):
        print(XXX.index(max(XXX))+1)
        print(max(XXX))
        XXX[XXX.index(max(XXX))] = -100


my_matrix = np.loadtxt(open("data.csv","rb"),delimiter=",",skiprows=0)

q = np.empty([1258,1], dtype = float)

#for i in range(2008):
#  subsum = 0
#  for j in range(50):
#    subsum = my_matrix[i*50+j][1] + subsum
#  q[i] = subsum/50
for i in range(1094):
    q[i] = my_matrix[i]



changepoint2(q)