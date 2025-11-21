import numpy as np
import matplotlib.pyplot as plt

def homography_estimate(x1, y1, x2, y2):
    return

def homography_apply(H, x1, y1):
    # extraire les coefs de H
    h11 = H[0][0]
    h12 = H[0][1]
    h13 = H[0][2]
    h21 = H[1][0]
    h22 = H[1][1]
    h23 = H[1][2]
    h31 = H[2][0]
    h32 = H[2][1]
    h33 = H[2][2]

    x2 = (h11 * x1 + h12 * y1 + h13)/(h31 * x1 + h32 * y1 + h33)
    y2 = (h21 * x1 + h22 * y1 + h23)/(h31 * x1 + h32 * y1 + h33)

    return (x2, y2)

def homography_extraction(I1, x, y, w, h):
    x2 = [0, 0, w, w]
    y2 = [0, h, 0, h]
    H = homography_estimate(x, y, x2, y2)
    I2 = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            I2[i][j] = homography_apply(H, x[i][j], y[i][j])
    return I2
