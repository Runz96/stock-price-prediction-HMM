import numpy as np
from keras.models import load_model

model = load_model('mlp_trained_model.h5') 

change = np.loadtxt(open("fixnumber.csv","rb"),delimiter=",",skiprows=0)


x_train = np.loadtxt(open("testdatanorm.csv","rb"),delimiter=",",skiprows=0)

x_predict = model.predict_classes(x_train)

solution = np.zeros([15840,2])
solution[:,1] = change
for i in range(len(solution)):
    solution[i][0] = int(x_predict[i])
    print(i)
    if solution[i][0] == 0:
        solution[i][1] = 0
        print(0)

np.savetxt('testsolution.csv', solution, delimiter = ',')

