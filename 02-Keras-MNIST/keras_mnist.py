"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

################################################################################
# %% IMPORT MNIST DATA
################################################################################

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

##### CONVERT DIMENSIONS FOR CONV2D
X_train = X_train[:,:,:,np.newaxis]/255
X_test = X_test[:,:,:,np.newaxis]/255

##### ONE HOT ENCODE
y_train_ohe = np.zeros(((y_train.size, y_train.max()+1)))
y_train_ohe[np.arange(y_train.size),y_train] = 1

y_test_ohe = np.zeros(((y_test.size, y_test.max()+1)))
y_test_ohe[np.arange(y_test.size),y_test] = 1

################################################################################
# %% BUILD MODEL
################################################################################

# DIMS: 32, 64, 128, 10 --> 0.9754 / 0.9861 / 4ep / 10min

model = keras.Sequential()
model.add(keras.layers.Conv2D(16, (3, 3), input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

################################################################################
# %% RUN MODEL
%%time
################################################################################

history = model.fit(
    x=X_train,
    y=y_train_ohe,
    epochs=10,
    verbose=1,
    validation_data=(X_test, y_test_ohe),
    #use_multiprocessing=True,
    batch_size=5000
)

#history.history['val_loss']
mp.semilogy(history.history['loss'], label='Training')
mp.semilogy(history.history['val_loss'], label='Testing')
mp.legend()
