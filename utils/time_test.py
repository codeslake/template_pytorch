import numpy as np
import cv2
from skimage import color
import time

a = (np.ones((2000, 2000, 3)) * 255).astype(np.uint8)

time_s = time.time()
a_cv2 = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
print(time.time() - time_s)

time_s = time.time()
a_sk = color.rgb2lab(a)
print(time.time() - time_s)

