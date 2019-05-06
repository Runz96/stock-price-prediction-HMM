import numpy as np


my_matrix = np.loadtxt(open("traindata.csv","rb"),delimiter=",",skiprows=0)
q = np.empty([1180,123], dtype = float)


for i in range(25,1205):
     for w in range(41):
         q[i-25][w*3]=my_matrix[i-20+w][0]
         q[i-25][w*3+1] = my_matrix[i-20+w][1]
         q[i-25][w*3+2] = my_matrix[i-20+w][2]

print(q)
np.savetxt('sum.csv',q,delimiter=',')