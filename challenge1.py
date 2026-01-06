import primitive as p
import matplotlib.pyplot as plt
import numpy as np


def count_pixel(image, colors):
    count = np.zeros(len(colors))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(len(colors)):
                if np.array_equal(image[i, j]*255, colors[k]):
                    count[k] += 1
    return count

def main():

    # importer l'image
    I = plt.imread("img/challenge1.png")
    h, w = I.shape[0], I.shape[1]

    # coordonnées du carré de référence
    carre_image_x = [591, 723, 617, 742]
    carre_image_y = [117, 144, 327, 325]

    # coordonnées du carré destination
    carre_ref_x = [300, 400, 300, 400]
    carre_ref_y = [50, 50, 150, 150]

    H = p.homography_estimate(carre_ref_x, carre_ref_y, carre_image_x, carre_image_y)

    new_w = 600
    new_h = 500

    # création de la nouvelle image
    new_I = np.ones((new_h, new_w, 3), dtype=I.dtype)
    for i in range(new_h):
        for j in range(new_w):
            x_i, y_i = p.homography_apply(H, j, i)
            if 0 <= round(x_i) < w and 0 <= round(y_i) < h:
                new_I[i, j] = I[int(round(y_i)), int(round(x_i))]

    # liste des couleurs
    colors = [[255, 0, 0], [255, 255, 0], [0, 255, 0], [255, 0, 255], [255, 192, 192], [192, 255, 192], [0, 0, 255], [192, 192, 255], [0, 0, 0]]
    colors_str = [("rouge",0.0), ("jaune",0.0), ("vert",0.0), ("violet",0.0), ("rose",0.0), ("vert pâle",0.0), ("bleu",0.0), ("gris",0.0), ("noir",0.0)]
    count = []

    # comptage des pixels
    max_size = 0
    max_index = 0
    count = count_pixel(new_I, colors)
    for i in range(len(count)):
        if count[i] > max_size:
            max_size = count[i]
            max_index = i

    # normalisation est tri dans l'ordre
    for i in range(len(count)):
        colors_str[i] = (colors_str[i][0], count[i]/max_size)
        
    colors_str.sort(key=lambda x: x[1], reverse=True)
    for i in range(len(colors_str)):
        print(f"{colors_str[i][0]} : {colors_str[i][1]}")

    plt.imshow(new_I)
    plt.show()

if __name__ == "__main__":
    main()
    
    