import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
from scipy.linalg import null_space

img_p = imread('./billboard_hacked_python.png')
print(img_p.shape)

img_c = imread('./billboard_hacked_cpp.png')
print(img_c.shape)

diff_l = []

for i in range(601):
    for j in range(900):
        for k in range(3):
            abs_err = abs(float(img_p[i][j][k]) - float(img_c[i][j][k]))
            if abs_err > 1:
                diff_l.append([i, j, k, abs_err, img_p[i][j][k], img_c[i][j][k]])

for pix in diff_l:
    print(pix)
