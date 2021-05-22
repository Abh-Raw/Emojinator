
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import pandas as pd

data = pd.read_csv("train_foo.csv")
data = data.dropna()
data.reset_index(inplace=True)
data.drop('index', axis='columns', inplace=True)            #Data cleaning
print(data)
dataset = np.array(data)            #shuffled data
np.random.shuffle(dataset)
X = dataset
Y = dataset
X = X[:, 1:2501]
Y = Y[:, 0]

X_train = X[0:16700, :]     #split into testing and training dataset
X_train = X_train / 255.
X_test = X[16700:, :]
X_test = X_test / 255.

Y = Y.reshape(Y.shape[0], 1)
Y_train = Y[0:16700, :]
Y_train = Y_train.T
Y_test = Y[16700:, :]
Y_test = Y_test.T

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print("Training samples = " + str(X_train.shape[0]))
print("Test samples = " + str(X_test.shape[0]))
print("X_train shape = " + str(X_train.shape))
print("Y_train shape = " + str(Y_train.shape))
print("X_test shape = " + str(X_test.shape))
print("Y_test shape = " + str(Y_test.shape))

image_x = 50
image_y = 50

le = LabelEncoder()

Y_train_new = le.fit_transform(Y_train.T)
Y_test_new = le.fit_transform(Y_test.T)
train_y = np_utils.to_categorical(Y_train_new)              #one hot encoding of categorical labels
test_y = np_utils.to_categorical(Y_test_new)
X_train = X_train.reshape(X_train.shape[0], image_x, image_y, 1)
X_test = X_test.reshape(X_test.shape[0], image_x,image_y, 1 )
X_train = np.asarray(X_train).astype(np.float64)
X_test = np.asarray(X_test).astype(np.float64)

print("X_train shape: " + str(X_train.shape))
print("X_test shape: " + str(X_test.shape))
print("train_y shape" + str(train_y.shape))

def keras_model(image_x, image_y):      #training model using CNN
    num_of_classes = 15
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size = (5, 5), strides = (5, 5), padding = 'same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))      #Used softmax activation for probability of each label
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "handEmo.h5"          #filepath for saved model in .h5 format
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_beat_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list

model, callbacks_list = keras_model(image_x, image_y)

# Fit the test data as vlidation in trained model
model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=1, batch_size=64, callbacks=callbacks_list)        #fitting model with data and validating with test data
scores = model.evaluate(X_test, test_y, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

#Printed model details
model.summary()             #printed the summary of model
model.save('handEmo.h5')        #saved model in .h5 format