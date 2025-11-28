import numpy as np
import matplotlib.pyplot as plt

def cross_product(A, B, C):
    # fait le produit vectoriel de AB et AC
    res = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
    return res

def is_in_quadrangle(X, Y, x, y):
    # X et Y sont les vecteurs des abscisses et ordonnées des points
    # on teste si P est dans le quadrangle ABCD /!\ doivent être dans l'ordre
    A = [X[0], Y[0]]
    B = [X[1], Y[1]]
    D = [X[2], Y[2]]
    C = [X[3], Y[3]]
    P = [x, y]

    cp1 = cross_product(A, B, P)
    cp2 = cross_product(B, C, P)
    cp3 = cross_product(C, D, P)
    cp4 = cross_product(D, A, P)

    cond = (cp1 >= 0 and cp2 >= 0 and cp3 >= 0 and cp4 >= 0) or (cp1 <= 0 and cp2 <= 0 and cp3 <= 0 and cp4 <= 0)
    return cond


def is_in_quadrangle_homography(H, x, y, size=100):
    u, v = homography_apply(H, x, y)
    return (0 <= u <= size) and (0 <= v <= size)


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
    x2 = [0, w, 0, w]
    y2 = [0, 0, h, h]
    H = homography_estimate(x2, y2, x, y)
    I2 = np.zeros((h, w, 3), dtype=I1.dtype)
    for i in range(h):
        for j in range(w):
            x_i, y_i = homography_apply(H, j, i)
            I2[i, j] = I1[int(round(y_i)), int(round(x_i))]
    return I2

def homography_cross_projection(I, x1, y1, x2, y2):

    h, w = I.shape[0], I.shape[1]
    I_res = I.copy()

    H21 = homography_estimate(x2, y2, x1, y1)
    H12 = homography_estimate(x1, y1, x2, y2)

    x_carre = [0, 100, 0, 100]
    y_carre = [0, 0, 100, 100]

    H1c = homography_estimate(x1, y1, x_carre, y_carre)
    H2c = homography_estimate(x2, y2, x_carre, y_carre)

    for i in range(h):
        for j in range(w):
            if is_in_quadrangle_homography(H1c, j, i, 100):
                x_i, y_i = homography_apply(H12, j, i)
                if 0 <= round(x_i) < w and 0 <= round(y_i) < h:
                    I_res[i, j] = I[int(round(y_i)), int(round(x_i))]
            if is_in_quadrangle_homography(H2c, j, i, 100):
                x_i, y_i = homography_apply(H21, j, i)
                if 0 <= round(x_i) < w and 0 <= round(y_i) < h:
                    I_res[i, j] = I[int(round(y_i)), int(round(x_i))]

    return I_res


def homography_projection(I1, I2, x, y):
    h_src, w_src = I1.shape[0], I1.shape[1]
    h_dst, w_dst = I2.shape[0], I2.shape[1]

    x_src = [0, w_src, 0, w_src]
    y_src = [0, 0, h_src, h_src]

    H = homography_estimate(x, y, x_src, y_src)
    I3 = I2.copy()

    for i in range(0, h_dst ):
        for j in range(0, w_dst ):
            x_i, y_i = homography_apply(H, j, i)
            if 0 <= round(x_i) < w_src and 0 <= round(y_i) < h_src:
                I3[i, j] = I1[round(y_i), round(x_i)]
                
    return I3

def main():

    # images
    # A -- B
    # |    |
    # C -- D
    image_tour = plt.imread('img/tour.jpg')

    points_x_tour_side = [123, 524, 131, 542]
    points_y_tour_side = [288, 96, 817, 740]

    image_grass = plt.imread('img/block_terre.jpeg')

    points_x_terre_side = [539, 955, 631, 957]
    points_y_terre_side = [281, 539, 704, 999]

    points_x_terre_top = [962, 1379, 540, 956]
    points_y_terre_top = [129, 284, 281, 539]

    image_tableaux = plt.imread('img/tableaux.jpg')

    points_x_tableaux1 = [762, 940, 755, 933]
    points_y_tableaux1 = [244, 236, 486, 506]

    points_x_tableaux2 = [93, 328, 94, 329]
    points_y_tableaux2 = [181, 212, 550, 520]


    ## Test de homography_extraction
    
    I2 = homography_extraction(image_grass, points_x_terre_side, points_y_terre_side, 400, 400)
    plt.imshow(I2)
    plt.show()

    ## Test de homography_cross_projection

    I2 = homography_cross_projection(image_grass, points_x_terre_side, points_y_terre_side, points_x_terre_top, points_y_terre_top)
    plt.imshow(I2)
    plt.show()

    I4 = homography_cross_projection(image_tableaux, points_x_tableaux1, points_y_tableaux1, points_x_tableaux2, points_y_tableaux2)
    plt.imshow(I4)
    plt.show()

    ## Test de homography_projection

    I3 = homography_projection(image_grass, image_tour, points_x_tour_side, points_y_tour_side)
    plt.imshow(I3)
    plt.show()

if __name__ == '__main__':
    main()
