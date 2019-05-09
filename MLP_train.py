import keras
import keras.backend as K
from keras.models import Sequential 
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop, SGD
from keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_train = np.loadtxt(open("sum.csv","rb"),delimiter=",",skiprows=0)
y_train = np.loadtxt(open("label.csv","rb"),delimiter=",",skiprows=0)
x_test = np.loadtxt(open("sum.csv","rb"),delimiter=",",skiprows=0)
y_test = np.loadtxt(open("label.csv","rb"),delimiter=",",skiprows=0)

model = Sequential()
model.add(Dense(1024,activation='relu',input_shape=(400,)))
#model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7,activation='softmax'))

model.summary()

def scheduler(epoch):
    if epoch == 40:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print('lr changed to', lr * 0.1)
    return K.get_value(model.optimizer.lr)

model.compile(loss='categorical_crossentropy',
#             optimizer=RMSprop(),
             optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0),
             metrics=['accuracy'])

reduce_lr = LearningRateScheduler(scheduler)

model.fit(x_train,y_train,batch_size=64,epochs=80,verbose=1,
         validation_data=(x_test,y_test), callbacks=[reduce_lr])


model.save('mlp_trained_model.h5')

score = model.evaluate(x_test,y_test,verbose=1)
print('Test loss:',score[0])
print('Test accuracy',score[1])

