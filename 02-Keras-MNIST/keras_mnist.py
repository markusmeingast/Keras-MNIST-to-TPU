"""
Implementation of MNIST CNN model using Keras, with the intention of running
TFLite model on Google Coral TPU Accelerator.
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

#from tensorflow import keras
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from time import time
from tensorflow.keras.callbacks import TensorBoard

################################################################################
# %% DEFINE FUNCTIONS / CLASSES
################################################################################



################################################################################
# %% IMPORT MNIST DATA
################################################################################

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

##### CONVERT DIMENSIONS FOR CONV2D
X_train = (X_train[:,:,:,np.newaxis])
X_test = (X_test[:,:,:,np.newaxis])

##### ONE HOT ENCODE
##### ON TRAINING DATA
y_train_ohe = np.zeros(((y_train.size, y_train.max()+1)))
y_train_ohe[np.arange(y_train.size),y_train] = 1

##### ON TESTING DATA
y_test_ohe = np.zeros(((y_test.size, y_test.max()+1)))
y_test_ohe[np.arange(y_test.size),y_test] = 1

################################################################################
# %% BUILD MODEL
################################################################################

##### GET INPUT SHAPE
(_, xdim, ydim, zdim) = X_train.shape

##### ADD LAYERS
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(xdim,ydim,zdim), dtype=tf.float32))
model.add(keras.layers.Conv2D(16, (3, 3)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10))#, activation='softmax'))

##### COMPILE MODEL AND PRINT SUMMARY
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

##### PRINT SUMMARY
print(model.summary())

################################################################################
# %% INIT TENSORBOARD
################################################################################

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

################################################################################
# %% RUN MODEL
#%%time
################################################################################

history = model.fit(
    x=X_train,
    y=y_train_ohe,
    epochs=1,
    verbose=1,
    validation_data=(X_test, y_test_ohe),
    use_multiprocessing=True,
    batch_size=5000,
    callbacks=[tensorboard]
)

model.save('keras_model.h5')

################################################################################
# %% PLOT LOSS HISTORY
################################################################################

mp.semilogy(history.history['loss'], label='Training')
mp.semilogy(history.history['val_loss'], label='Testing')
mp.legend()
mp.show()

################################################################################
# %% CONVERT
################################################################################

def representative_dataset_gen():
    for i in range(100):
        yield [X_train[i, None].astype(np.float32)]

##### CREATE CONVERTER
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('keras_model.h5')

##### QUANTIZE MODEL
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

#converter.representative_dataset = representative_dataset_gen

converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.inference_type = tf.float32
converter.std_dev_values = 0.3
converter.mean_values = 0.5
converter.default_ranges_min = 0.0
converter.default_ranges_max = 1.0

converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                       tf.lite.OpsSet.SELECT_TF_OPS]

converter.post_training_quantize=True

##### CONVERT THE MODEL
tflite_model = converter.convert()
##### SAVE MODEL TO FILE
tflite_model_name = "mnist.tflite"
open(tflite_model_name, "wb").write(tflite_model)

#    --input_arrays=conv2d_input \
#    --output_arrays=dense/Softmax \

"""


converter.inference_type=tf.float32
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

##### SET DEFAULT OPTIMIZATIONS

#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
##### REDUCE/OPTIMIZE MODEL --> SLIGHT PENALTY IN MODEL ACCURACY

"""


################################################################################
# %% VALIDATE
################################################################################

"""
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mymodel.tflite")
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
"""


X_train.max()
