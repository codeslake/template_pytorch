import cv2
import numpy as np

l_max = -100
a_max = -100
b_max = -100
l_min = 1000
a_min = 1000
b_min = 1000

h_max = -100
s_max = -100
v_max = -100
h_min = 1000
s_min = 1000
v_min = 1000

for r in np.arange(255):
    for g in np.arange(255):
        for b in np.arange(255):
            pixel_bgr = np.array([b, g, r]).astype(np.uint8)
            #pixel_bgr = np.array([b, g, r]).astype(np.float32)/255.
            pixel_bgr = np.expand_dims(pixel_bgr, axis = 0)
            pixel_bgr = np.expand_dims(pixel_bgr, axis = 0)

            pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2LAB)
            pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)

            l_max = max(l_max, pixel_lab[0][0][0])
            a_max = max(a_max, pixel_lab[0][0][1])
            b_max = max(b_max, pixel_lab[0][0][2])
            l_min = min(l_min , pixel_lab[0][0][0])
            a_min = min(a_min , pixel_lab[0][0][1])
            b_min = min(b_min , pixel_lab[0][0][2])

            h_max = max(h_max, pixel_hsv[0][0][0])
            s_max = max(s_max, pixel_hsv[0][0][1])
            v_max = max(v_max, pixel_hsv[0][0][2])
            h_min = min(h_min , pixel_hsv[0][0][0])
            s_min = min(s_min , pixel_hsv[0][0][1])
            v_min = min(v_min , pixel_hsv[0][0][2])


print(l_max, a_max, b_max)
print(l_min, a_min, b_min)

print(h_max, s_max, v_max)
print(h_min, s_min, v_min)

