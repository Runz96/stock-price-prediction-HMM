import numpy as np
from numpy import random
import cmath
import os

iter = 0

#low dimentional change point
#
#this is the code for one variation
def changepoint2(q):
    global iter
    n = len(q)
    XXX = []
    for i in range(n):
        if i ==0:
            Wk = q[0]
        else:
            Wk = np.sum(q[0:i])
        sum2 = np.sum(q)
        summ = Wk/pow(n,0.5)-(i+1)*sum2/pow(n,1.5)
        Sum = abs(summ)
        K = pow(2000*(i+1)*(1-(i+1)/n)/n,0.5)*5
        ##K = pow((i+1)*(1-(i+1)/n)/n,0.5)*2000
        KK = Sum/K
        XXX.append(KK)
    del XXX[-1]
    #print(XXX)
    #print(time)
  
    #for i in range(10):
    while True: 
    #print(XXX.index(max(XXX))+1)
        #print(max(XXX))


        #if abs(XXX.index(max(XXX))-train_answer)<200:
         #if abs(XXX[XXX.index(max(XXX))]-XXX[XXX.index(max(XXX))-1])>0.05 and abs(XXX[XXX.index(max(XXX))]-XXX[XXX.index(max(XXX))+1])>0.05 and abs(XXX[XXX.index(max(XXX))+1]-XXX[XXX.index(max(XXX))-1])<0.5 and time[XXX.index(max(XXX))]/1000000>30:
         if time[XXX.index(max(XXX))]/1000000>30 and time[XXX.index(max(XXX))]/1000000<400 :   
            print(i+1)
            #print("which time square")
            print(XXX.index(max(XXX))+1) 
            print(max(XXX))
            print(time[XXX.index(max(XXX))]/1000000)
            print("\n")
            break
         XXX[XXX.index(max(XXX))] =np.array([-100+0.j])

#filename = "104915.csv"
#train_answer = 1500
#my_matrix = np.loadtxt(open(filename,"rb"),delimiter=",",skiprows=0)


answer_matrix = np.zeros([120,1])

for info in os.listdir(r'E:\project\testing\sss'):
  domain = os.path.abspath(r'E:\project\testing\sss')
#for ss in range(len(w)):
  info = os.path.join(domain,info)          
  
  my_matrix = np.loadtxt(open(info,"rb"),delimiter=",",skiprows=0)
# q = np.zeros(0)  

# for i in range(len(my_matrix)//50):
#     subsum = 0
#     for j in range(50):
#         subsum = my_matrix[i*50+j][1] + subsum
#     q = np.append(q,subsum/50)


#partition_num configuration
  partition_num = 50
  q_num = len(my_matrix)//partition_num
  q = np.zeros(q_num)  
  time = np.zeros(q_num)

  for i in range(q_num):
    subsum = 0
    subsum2 = 0
  #here define how to get the feature
    for j in range(partition_num):
      subsum = my_matrix[i*partition_num+j][1] + subsum
      subsum2 = my_matrix[i*partition_num+j][0] + subsum2
    q[i] = subsum/partition_num
    if i ==0:
        time[i] = subsum2
    else:
        time[i] = subsum2+time[i-1]


  changepoint2(q)