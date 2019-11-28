"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.classification.engine import ClassificationEngine
import time
import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as mp

################################################################################
# %% DEFAULT SETTINGS
################################################################################

np.set_printoptions(precision=3)

################################################################################
# %% DEFINE FUNCTIONS
################################################################################

################################################################################
# %% IMPORT MODEL AND MOVE TO TPU
################################################################################

engine = BasicEngine('mnist_edgetpu.tflite')
(_, xdim, ydim, zdim) = engine.get_input_tensor_shape()

################################################################################
# %% INIT SCREEN OUTPUT
################################################################################

cap = cv2.VideoCapture(0)

mp.figure()

################################################################################
# %% RUN MODEL OFF OF CAMERA ON TPU
################################################################################
it = 0
while cap.isOpened():

    ##### GRAB IMAGE FROM CAM
    ret, frame = cap.read()
    if not ret:
        break
    image = frame

    ##### RESIZE TO INPUT TENSOR SHAPE
    image = cv2.resize(image, (xdim, ydim))

    ##### CONVERT TO GRAYSCALE
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ##### INVERT
    image = cv2.bitwise_not(image)
    _, image = cv2.threshold(image,180,255,cv2.THRESH_BINARY)

    ##### FLATTEN INPUT
    input = image.flatten()

    ##### RUN ON TPU
    results = engine.run_inference(input)

    ##### PRINT RESULTS
    #if it % 20 == 0:
    #    print(' | 0 | | 1 | | 2 | | 3 | | 4 | | 5 | | 6 | | 7 | | 8 | | 9 | ')
    #print(engine.get_raw_output())
    mp.gca().cla()
    mp.bar(np.arange(10),engine.get_raw_output())
    mp.axis([-0.5,9.5,0,1])
    mp.pause(0.01)

    image = cv2.resize(image, (560, 560))

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    it += 1

cap.release()
cv2.destroyAllWindows()
