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

np.set_printoptions(precision=3)

################################################################################
# %% DEFINE FUNCTIONS
################################################################################

################################################################################
# %% IMPORT MODEL AND MOVE TO TPU
################################################################################

engine = BasicEngine('mnist_edgetpu.tflite')
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
(_, xdim, ydim, zdim) = engine.get_input_tensor_shape()

################################################################################
# %% INIT SCREEN OUTPUT
################################################################################

cap = cv2.VideoCapture(0)

################################################################################
# %% RUN MODEL OFF OF CAMERA ON TPU
################################################################################

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

    ##### FLATTEN INPUT
    input = (image.flatten()/64).astype(np.uint8)

    #engine.RunInference(tensor)
    #results = engine.ClassifyWithInputTensor(input)
    results = engine.run_inference(input)
    print(f'{results[1]}')
    #engine.get_raw_output()

    #cv2_im = append_objs_to_img(cv2_im, objs, labels)

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


engine.get_all_output_tensors_sizes()
engine.total_output_array_size()
