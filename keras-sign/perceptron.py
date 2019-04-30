# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 10

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()

img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# you may want to normalize the data here..

# normalize data
X_train = X_train.astype('float32') / 255. # keras will train better if normalized
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# create model, the images are 28x28
model = Sequential() # using Model() will be better for more complicated topology
#model.add(Flatten(input_shape=(img_width, img_height))) # flatten into single vector, learning zero parameters
model.add(Conv2D(32, (3,3), input_shape=(img_width, img_height, 1))) # slides 32 kernels of 3x3 over the image, needed to remove flatten because needs 2-D array, but need to flatten after this convolution
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3))) # slides 32 kernels of 3x3 over the image, needed to remove flatten because needs 2-D array, but need to flatten after this convolution
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3))) # slides 32 kernels of 3x3 over the image, needed to remove flatten because needs 2-D array, but need to flatten after this convolution
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(100, activation='relu')) # 1000 classes with relu activation because default in keras is linear (more classes will complicate the model, will run slower and may overfit)
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2)) # sets 20% of output to zero, will be randomly selected
model.add(Dense(num_classes, activation="softmax")) # will output the number of classes, learning 784 (28x28) parameters (multiply each weight by each input) and then sum
model.compile(loss=config.loss, optimizer=config.optimizer,
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])
