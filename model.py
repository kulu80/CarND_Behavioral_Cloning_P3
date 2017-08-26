## This part will laod neccessary modules and access the dataset 
## and save the image data and the steering angle measurment as
## python lists
import csv
import numpy as np
import cv2
import matplotlib.image as mpimg
images = []
measurements = []

with open('/home/bantwale/data1/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        measurement = float(line[3])
        measurements.append(measurement)
        correction = 0.2
        measurements.append(measurement+correction)
        measurements.append(measurement-correction)
        for i in range(3):
            source_path = line[0]
            file_name = source_path.split('/')[-1]
            current_path = './data1/IMG/' +file_name
            image = cv2.imread(current_path)
            images.append(image)
# this section makes new lists by taking the original
# listed created above and agumented the data (1). by
# flipping the image horizontally and (2). by multiplying measurements 
# by -1. Preprocessing package 'StandardScaler' from sklearn is appplied to
# standardaize the steering angle 

augmented_images = []
augmented_measurements = []
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image,1)
    augmented_images.append(flipped_image)
    flipped_measurement = float(measurement)*-1.0
    augmented_measurements.append(flipped_measurement)
            
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements) 

from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#y_train = min_max_scaler.fit_transform(y_train)

scaler = preprocessing.StandardScaler(with_mean = False).fit(y_train)
y_train_scaled = scaler.transform(y_train) 
#y_train_scaled = y_train_scaled.reshape(-1,1)
#y_train_scaled.shape
# this is the main model framework which is the same as
# NVIDIA's model 
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout
from keras.layers.convolutional import Conv2D,Cropping2D
from keras.layers import MaxPool2D

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape =(160,320,3)))
model.add(Cropping2D(cropping =((70,25),(0,0)),input_shape =(160,320,3)))
model.add(Dropout(0.2,input_shape=(160,320,3)))
model.add(Conv2D(24,5,5,subsample=(2,2),activation = 'relu'))
#model.add(MaxPool2D())
model.add(Conv2D(36,5,5,subsample=(2,2),activation = 'relu'))
#model.add(MaxPool2D())
model.add(Conv2D(48,5,5,subsample=(2,2),activation = 'relu'))
model.add(Conv2D(64,3,3,activation = 'relu'))
#model.add(MaxPool2D())
model.add(Conv2D(64,3,3,activation = 'relu'))
#model.add(MaxPool2D())
#model.add(MaxPool2D())
#model.add(Convolution2D(16,5,5,activation='relu'))
#model.add(MaxPool2D())
model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer = 'adam')
model.fit(X_train,y_train_scaled,validation_split=0.2,shuffle=True,epochs =2)
model.save('model_nvidia_6.h5')

