"""
Implementation of FFNN classification with Keras frontend and TF backend.
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

from tensorflow import keras
from sklearn.datasets import make_moons
import matplotlib.pyplot as mp

################################################################################
# %% CREATE DATASETS
################################################################################

X_train, y_train = make_moons(n_samples=500, noise=0.1, random_state=42)
X_test, y_test = make_moons(n_samples=50, noise=0.1, random_state=28)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

################################################################################
# %% CREATE MODELS
################################################################################

model = keras.Sequential()
model.add(keras.layers.Dense(5, input_shape=(X_train.shape[1],)))
model.add(keras.layers.Dense(3))
model.add(keras.layers.Dense(2))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################################################################
# %% RUN MODEL
################################################################################

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    verbose=2,
    validation_data=(X_test, y_test),
    use_multiprocessing=True
)

################################################################################
# %% PLOT GRAPH
################################################################################

mp.semilogy(history.history['loss'], label='Training')
mp.semilogy(history.history['val_loss'], label='Testing')
mp.legend()
