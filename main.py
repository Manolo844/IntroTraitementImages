import numpy as np
import matplotlib.pyplot as plt

import primitive as p
import mosaique as m

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

    image_ville1 = plt.imread('img/mosaique_ville/image1.jpg')
    image_ville2 = plt.imread('img/mosaique_ville/image2.jpg')
    image_ville3 = plt.imread('img/mosaique_ville/image3.jpg')
    image_ville4 = plt.imread('img/mosaique_ville/image4.jpg')

    points_x_ville12 = [732, 867, 528, 782]
    points_y_ville12 = [719, 1070, 1153, 1295]

    points_x_ville21 = [229, 499, 227, 516]
    points_y_ville21 = [943, 1206, 1422, 1445]

    points_x_ville23 = [948, 1222, 962, 1238]
    points_y_ville23 = [1032, 1008, 1398, 1381]

    points_x_ville32 = [557, 602, 194, 231]
    points_y_ville32 = [65, 334, 106, 380]

    points_x_ville34 = [1132, 1184, 786, 872]
    points_y_ville34 = [1013, 1215, 957, 1376]

    points_x_ville43 = [80, 90, 395, 434]
    points_y_ville43 = [1469, 1261, 1623, 1198]

    ## Test de homography_extraction
    
    I2 = p.homography_extraction(image_grass, points_x_terre_side, points_y_terre_side, 400, 400)
    plt.imshow(I2)
    plt.title("Test pour extraction")
    plt.show()

    ## Test de homography_cross_projection

    I2 = p.homography_cross_projection(image_grass, points_x_terre_side, points_y_terre_side, points_x_terre_top, points_y_terre_top)
    plt.imshow(I2)
    plt.title("Test pour cross projection")
    plt.show()

    I4 = p.homography_cross_projection(image_tableaux, points_x_tableaux1, points_y_tableaux1, points_x_tableaux2, points_y_tableaux2)
    plt.imshow(I4)
    plt.title("Test pour cross projection")
    plt.show()

    ## Test de homography_projection

    I3 = p.homography_projection(image_grass, image_tour, points_x_tour_side, points_y_tour_side)
    plt.imshow(I3)
    plt.title("Test pour projection")
    plt.show()

    ## Test de mosaique

    mib1 = m.i_to_mib(image_ville1)
    mib2 = m.i_to_mib(image_ville2)
    mib3 = m.i_to_mib(image_ville3)
    mib4 = m.i_to_mib(image_ville4)
    list_mib = [mib1, mib2, mib3, mib4]

    H12 = p.homography_estimate(points_x_ville12, points_y_ville12, points_x_ville21, points_y_ville21)
    H23 = p.homography_estimate(points_x_ville23, points_y_ville23, points_x_ville32, points_y_ville32)
    H34 = p.homography_estimate(points_x_ville34, points_y_ville34, points_x_ville43, points_y_ville43)
    
    H3_to_4 = H34
    H2_to_4 = H3_to_4 @ H23
    H1_to_4 = H2_to_4 @ H12
    
    

    print("Transformation image 1")
    mib1_transform = m.mib_transform(mib1, H1_to_4)
    print("Transformation image 2")
    mib2_transform = m.mib_transform(mib2, H2_to_4)
    print("Transformation image 3")
    mib3_transform = m.mib_transform(mib3, H3_to_4)
    mib4_transform = mib4

    image1_result = mib1_transform[1]
    image2_result = mib2_transform[1]
    image3_result = mib3_transform[1]
    image4_result = mib4_transform[1]

    image_global = m.mib_fusion([mib1_transform, mib2_transform, mib3_transform, mib4_transform])[1]

    # affichage

    plt.imshow(image_global)
    plt.title("Test pour mosaique : Image globale")
    plt.show()

    figure = plt.figure(figsize=(10, 10))
    figure.add_subplot(2, 2, 1)
    plt.title("Image 1 projettée sur 4")
    plt.imshow(image1_result)
    
    figure.add_subplot(2, 2, 2)
    plt.title("Image 2 projettée sur 4")
    plt.imshow(image2_result)
    
    figure.add_subplot(2, 2, 3)
    plt.title("Image 3 projettée sur 4")
    plt.imshow(image3_result)
    
    figure.add_subplot(2, 2, 4)
    plt.title("Image 4 (Référence)")
    plt.imshow(image4_result)
    
    plt.show()

if __name__ == '__main__':
    main()
