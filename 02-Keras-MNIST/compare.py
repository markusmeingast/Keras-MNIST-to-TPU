import numpy as np
import matplotlib.pyplot as mp

keras_test = np.array([6.2409010e-05, 5.4695935e-04, 2.5008904e-04, 2.5657348e-03,
        9.5104867e-01, 6.7629799e-04, 2.7148254e-04, 1.9922120e-02,
        2.5084873e-03, 2.2147831e-02], dtype=np.float32)


keras_test = np.array([7.8120956e-04, 6.8606285e-04, 1.1962154e-03, 6.0702059e-03,
        5.1717274e-02, 1.5469956e-03, 6.9667405e-04, 3.7053298e-02,
        4.1750683e-03, 8.9607692e-01], dtype=np.float32)

edgetpu_test = np.array([0.   , 0.   , 0.   , 0.004, 0.953, 0.   , 0.   , 0.02 , 0.004,
        0.02 ], dtype=np.float32)

edgetpu_test = np.array([0.   , 0.   , 0.   , 0.008, 0.047, 0.   , 0.   , 0.035, 0.004,
        0.898], dtype=np.float32)

fig, ax = mp.subplots()
ax.bar(np.arange(10),keras_test,width = 0.35)
ax.bar(np.arange(10)+0.35,edgetpu_test,width = 0.35)
