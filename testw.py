import numpy as np
from numpy import random
import cmath
import os

#low dimentional change point
#
#this is the code for one variation
def changepointrealm(q):
   n = len(q)
   p = len(q[0])
   XXX = [0]*n
   XX = []
   for i in range(n):
       D = np.ones((p,p))*0
       if i == 0:
           aa = q[0,:]
           bb = q[1:n,:]
       else:
           aa = q[0:i,:]
           bb = q[i+1:n,:]
       aa1 = np.cov(aa.T)
       bb1 = np.cov(bb.T)
       Sk = (aa1 * (i+1)+bb1*(n-i-1))/(n-2)
       summ = np.diag(Sk)
       for w in range(p):
           D[w,w]=summ[w]
       DD = np.linalg.inv(D)
       sum1= aa.sum(axis =0)
       sum2= q.sum(axis = 0)*(i+1)/n
       ss = sum1-sum2
       www= ss.reshape(ss.shape[0],1)
       www1=www.T
       DD1 =(www1/n/cmath.sqrt(p))@DD@www
       DD2 = i*(n-i)*cmath.sqrt(p)*(1+2/n)/pow(n,2)
       x = DD1-DD2
       XXX[i]=x[0]
   for h in range(3,n-3):
       XX.append((XXX[h]+XXX[h-1]+XXX[h+1])/3)
  # XX = np.array(XX).reshape((-1))
   print(XX)
   #nds = np.argsort(XX)[::-1]
   #print(nds)
   #print (XX)
   
   for i in range(10):
        #print(XXX.index(max(XXX))+1)
        #print(max(XXX))


        #if abs(XXX.index(max(XXX))-train_answer)<200:
         #if abs(XX[XX.index(max(XX))]-XX[XX.index(max(XX))-1])>0.05 and abs(XX[XX.index(max(XX))]-XX[XX.index(max(XX))+1])>0.05 and abs(XX[XX.index(max(XX))+1]-XX[XX.index(max(XX))-1])<0.5 and time[XX.index(max(XX))]/1000000>30:
         #if time[XX.index(max(XX))]/1000000>30 and time[XX.index(max(XX))]/1000000<400 :
         print(i+1)
            #print("which time square")
         print(XX.index(max(XX))+4)
         print(max(XX))
         #print(time[XX.index(max(XX))+4]/1000000)
         print("\n")
         XX[XX.index(max(XX))] =np.array([-100+0.j])
            


         
filename = "data.csv"
train_answer = 550
my_matrix = np.loadtxt(open(filename,"rb"),delimiter=",",skiprows=0)


# q = np.zeros(0)  

# for i in range(len(my_matrix)//50):
#     subsum = 0
#     for j in range(50):
#         subsum = my_matrix[i*50+j][1] + subsum
#     q = np.append(q,subsum/50)


#partition_num configuration
partition_num = 5
q_num = len(my_matrix)//partition_num

# ##only one feature
# q = np.zeros(q_num)  

# for i in range(q_num):
#   subsum = 0
#   #here define how to get the feature
#   for j in range(partition_num):
#     subsum = my_matrix[i*partition_num+j][1] + subsum
#   q[i] = subsum/partition_num

##multi feature
feature_num = 4
q = np.zeros([q_num,feature_num])
time = np.zeros(q_num)
for i in range(q_num):
    median_matrix = np.zeros(0)
    subsum1 = 0
    subsum2 = 0
    subsum3 = 0
    subsum4 = 0
    #here define how to get the feature
    for j in range(partition_num):
        subnumber = my_matrix[i*partition_num+j]
        subsum1 = subnumber + subsum1
        subsum2 = subnumber*subnumber + subsum2
        subsum3 = subnumber*subnumber*subnumber + subsum3
        median_matrix = np.append(median_matrix,subnumber)
        #subsum4 = my_matrix[i*partition_num+j][0] + subsum4
    q[i][0] = subsum1/partition_num
    q[i][1] = subsum2/partition_num
    q[i][2] = subsum3/partition_num
    q[i][3] = np.median(median_matrix)
    #if i ==0:
    #  time[i] = subsum4
    #else:
    #  time[i] = subsum4+time[i-1]

changepointrealm(q)
