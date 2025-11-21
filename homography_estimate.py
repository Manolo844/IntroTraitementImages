import numpy as np
import matplotlib.pyplot as plt

def homography_estimate(x1, y1, x2, y2):
    H = np.array([[0,0,0],[0,0,0],[0,0,1]])
    A=np.zeros((8,8))
    B=np.zeros(8)

    for i in range(len(x1)):
        A[2*i] = [x1[i], y1[i], 1, 0, 0, 0, -x1[i]*x2[i], -y1[i]*x2[i]]
        A[2*i+1] = [0, 0, 0, x1[i], y1[i], 1, -x1[i]*y2[i], -y1[i]*y2[i]]

    for i in range(len(x2)):
        B[2*i] = x2[i]
        B[2*i+1] = y2[i]

    h = np.linalg.solve(A, B)
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1]
    ])

    return H

def homography_apply(H, x1, y1):
    # extraire les coefs de H
    h11, h12, h13 = H[0][0], H[0][1], H[0][2]
    h21, h22, h23 = H[1][0], H[1][1], H[1][2]
    h31, h32, h33 = H[2][0], H[2][1], H[2][2]

    x2 = (h11 * x1 + h12 * y1 + h13)/(h31 * x1 + h32 * y1 + h33)
    y2 = (h21 * x1 + h22 * y1 + h23)/(h31 * x1 + h32 * y1 + h33)

    return (x2, y2)

def homography_extraction(I1, x, y, w, h):
    borne_I1 = I1.shape
    x2 = [0, w, 0, w]
    y2 = [0, 0, h, h]
    H = homography_estimate(x2, y2, x, y)
    I2 = np.zeros((h, w, 3), dtype=I1.dtype)
    for i in range(h):
        for j in range(w):
            x_i, y_i = homography_apply(H, j, i)
            I2[i, j] = I1[int(round(y_i)), int(round(x_i))]
    return I2

def main():
    image_tour = plt.imread('img/tour.jpg')
    # A -- B
    # |    |
    # C -- D
    points_x = [122, 524, 130, 542]
    points_y = [289, 96, 817, 737]

    I2 = homography_extraction(image_tour, points_x, points_y, 2000, 2000)
    plt.imshow(I2)
    plt.show()

if __name__ == '__main__':
    main()
