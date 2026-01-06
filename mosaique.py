import numpy as np
import matplotlib.pyplot as plt
import primitive as p

def i_to_mib(I):
    h, w = I.shape[0], I.shape[1]
    mask = np.ones((h, w))
    image = I.copy()
    boite = [(0,0), (w, h)]
    return (mask, image, boite)

def mib_transform(mib, H):
    # extraction des composantes
    mask = mib[0]
    image = mib[1]
    h_src, w_src = mib[2][1][1] - mib[2][0][1], mib[2][1][0] - mib[2][0][0]

    # calcul de la nouvelle boite englobante
    corners_x = [0, w_src, 0, w_src]
    corners_y = [0, 0, h_src, h_src]

    new_corners_x = []
    new_corners_y = []

    for i in range(4):
        x, y = p.homography_apply(H, corners_x[i], corners_y[i])
        new_corners_x.append(x)
        new_corners_y.append(y)

    min_x = min(new_corners_x)
    max_x = max(new_corners_x)
    min_y = min(new_corners_y)
    max_y = max(new_corners_y)

    # offset = coin haut-gauche
    offset_x = int(np.floor(min_x))
    offset_y = int(np.floor(min_y))

    # taille nouvelle image
    new_w = int(np.ceil(max_x)) - offset_x
    new_h = int(np.ceil(max_y)) - offset_y

    new_border = [(offset_x, offset_y), (offset_x + new_w, offset_y + new_h)]

    # initialisation masque et image
    new_mask = np.zeros((new_h, new_w))
    new_image = np.zeros((new_h, new_w, 3), dtype=image.dtype)

    # creation masque et image
    H_inv = np.linalg.inv(H)

    for i in range(new_h):
        for j in range(new_w):
            # coordonnée réelle dans le référentiel global
            real_x = j + offset_x
            real_y = i + offset_y

            # chercher d'où ça vient dans l'image source
            src_x, src_y = p.homography_apply(H_inv, real_x, real_y)
            
            src_x_int = int(round(src_x))
            src_y_int = int(round(src_y))

            # vérification 
            if 0 <= src_x_int < w_src and 0 <= src_y_int < h_src:
                new_image[i, j] = image[src_y_int, src_x_int]
                new_mask[i, j] = mask[src_y_int, src_x_int]

    return (new_mask, new_image, new_border)

def mib_fusion_couple(mib1, mib2):
    mask1, img1, border1 = mib1
    mask2, img2, border2 = mib2

    min_x = min(border1[0][0], border2[0][0])
    min_y = min(border1[0][1], border2[0][1])
    max_x = max(border1[1][0], border2[1][0])
    max_y = max(border1[1][1], border2[1][1])

    final_img = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.float32)
    final_mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.float32)

    for y_global in range(min_y, max_y):
        for x_global in range(min_x, max_x):
            pixel_sum = np.zeros(3)
            weight_sum = 0          

            #verifier si dans image 1
            y1_local = y_global - border1[0][1]
            x1_local = x_global - border1[0][0]

            if 0 <= x1_local < img1.shape[1] and 0 <= y1_local < img1.shape[0]:
                if mask1[y1_local, x1_local] == 1:
                    pixel_sum += img1[y1_local, x1_local]
                    weight_sum += 1
            
            #verifier si dans image 2
            y2_local = y_global - border2[0][1]
            x2_local = x_global - border2[0][0]

            if 0 <= x2_local < img2.shape[1] and 0 <= y2_local < img2.shape[0]:
                if mask2[y2_local, x2_local] == 1:
                    pixel_sum += img2[y2_local, x2_local]
                    weight_sum += 1

            y_final = y_global - min_y
            x_final = x_global - min_x
            
            #si dans les deux images
            if weight_sum > 0:
                final_img[y_final, x_final] = pixel_sum / weight_sum
                final_mask[y_final, x_final] = 1

    final_border = [(min_x, min_y), (max_x, max_y)]
    
    return (final_mask, final_img.astype(np.uint8), final_border)

def mib_fusion(list_mib):
    mib_global = list_mib[0]
    for mib in list_mib[1:]:
        mib_global = mib_fusion_couple(mib_global, mib)
    return mib_global