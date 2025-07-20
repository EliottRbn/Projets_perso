# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 22:51:56 2025

@author: eliot
"""

import numpy as np
import cv2 
import os
import pickle
import numba 
from math import cos, sin, radians

files = [file for file in os.listdir('BD/train') if file == "neutral"]
pictures = os.listdir(f"BD/train/{files[0]}")
pict_test = cv2.imread(f"BD/train/{files[0]}/{pictures[0]}", 0)
pict_test = cv2.resize(pict_test, dsize = (32,32)).astype("float64")

cv2.imshow('test',pict_test.astype("uint8"))

#%% Test d'un code pour features de HOG : programmé dans le but de pouvoir appliquer numba dessus, on ne cherche pas la simplicité algorithmique mais l'efficacité

import time 

def timer(funct):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = funct(*args, **kwargs)
        end = time.time()
        print(f"Temps d'exécution : {end - start:.4f}s")
        return result
    return wrapper

@numba.njit
def hist_OG(G, theta, n, nbins = 9, min_val = 0, max_val = 180):
    hist = np.zeros(nbins, dtype=float)
    bin_width = (max_val - min_val) / nbins
    for i in range(n):
        for j in range(n):
            val = theta[i,j]
            if min_val <= val < max_val:
                bin_index = int((val - min_val) / bin_width)
                if bin_index >= nbins:  # pour éviter arrondi bizarre
                    bin_index = nbins - 1
                hist[bin_index] += G[i,j]
    return hist 

@numba.njit
# @timer
def HOG(pict):
    
    ### Step 1 : Normalisation via la racine des pixels ###
    
    n = pict.shape[0]
    pict = np.sqrt(pict)
    
    ### Step 2 : Calcul du gradient et des orientations ### 
    
    Dy = np.zeros((n, n))
    Dx = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Diagonale supérieure
            if j == i + 1:
                Dy[i,j] = 1
                Dx[j,i] = 1
            # Diagonale inférieure
            if j == i-1 : 
                Dy[i,j] = -1 
                Dx[j,i] = -1
    
    Dy = Dy/2 
    Dx = Dx/2
    
    # Conditions au bord 
    
    Dy[0,0] = -0.5 
    Dy[n-1,n-1] = 0.5 
    
    Dx[0,0] = -0.5 
    Dx[n-1,n-1] = 0.5
    
    # Gradients & orientations
    
    Gx = pict@Dx
    Gy = Dy@pict
    
    G = np.sqrt(Gx**2 + Gy**2)
    
    theta = np.degrees(np.arctan2(Gy,Gx)) % 180
    
    HOG_features = []
    
    ### Step 3 : Weigthed vote in each cell ###
    
    nbins = 9
    cell_size = 8 # On fait des cellules de taille 8x8 soit 64 cellules 
    fraction_size = n//cell_size
    cell_storage = np.empty((nbins*fraction_size,fraction_size), dtype = float)
    
    for j in range(fraction_size):
        for i in range(fraction_size):
            window_G = G[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size]
            window_theta = theta[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size]
            cell_storage[i*nbins:(i+1)*nbins,j] = hist_OG(window_G,window_theta, cell_size)
    
    ### Test pour l'affichage ### 

    # hog_image = np.zeros((n, n), dtype=float)
    # angle_step = 180 // nbins
    # half_cell = cell_size // 2

    # n_cells = n // cell_size

    # for y in range(n_cells):
    #     for x in range(n_cells):
    #         center_y = y * cell_size + half_cell
    #         center_x = x * cell_size + half_cell
    #         hist = cell_storage[(x*nbins):(x+1)*nbins, y]

    #         for bin_idx in range(nbins):
    #             angle = bin_idx * angle_step
    #             rad = radians(angle)
    #             magnitude = hist[bin_idx]

    #             dx = cos(rad) * half_cell * magnitude
    #             dy = sin(rad) * half_cell * magnitude

    #             y1 = int(center_y - dy)
    #             y2 = int(center_y + dy)
    #             x1 = int(center_x - dx)
    #             x2 = int(center_x + dx)

    #             # On vérifie qu'on est dans les limites de l'image
    #             if 0 <= x1 < n and 0 <= x2 < n and 0 <= y1 < n and 0 <= y2 < n:
    #                 rr, cc = draw_line(y1, x1, y2, x2, n)
    #                 hog_image[rr, cc] += magnitude  # Ajouter l’intensité

    # # Normalisation pour affichage
    # hog_image = hog_image / np.max(hog_image)
    
    ### Step 4 : Blocks normalization ### -> On prend des blocs de 2x2 cellules 
    
    for j in range(fraction_size - 1):
        for i in range(fraction_size - 1):
            concatenate = np.empty(4*nbins)
            concatenate[:nbins] = cell_storage[i*nbins:(i+1)*nbins,j] 
            concatenate[nbins:2*nbins] = cell_storage[(i+1)*nbins:(i+2)*nbins,j] 
            concatenate[2*nbins:3*nbins] = cell_storage[i*nbins:(i+1)*nbins,j+1]
            concatenate[3*nbins:4*nbins] = cell_storage[(i+1)*nbins:(i+2)*nbins, j+1]
            # HOG_features.append(concatenate / np.sqrt(concatenate.T@concatenate)) 
            
            # Test de la normalisation L2-Hys : à voir si on a des meilleurs perfs avec ça ou pas 
            
            norm = np.sqrt(np.sum(concatenate ** 2))
            if norm != 0.0: # Éviter les divisions par zéro et donc les blocks entiers de nan
                concatenate = concatenate / norm 
            
            concatenate[concatenate > 0.2] = 0
            
            norm = np.sqrt(np.sum(concatenate ** 2))
            if norm != 0.0: # Éviter les divisions par zéro et donc les blocks entiers de nan
                concatenate = concatenate / norm
                
            HOG_features.append(concatenate)
    
    nb_features = len(HOG_features)
    shape_features = HOG_features[0].shape[0]
    
    hog_features = np.empty(nb_features*shape_features, dtype = float)
    
    for i,feature in enumerate(HOG_features):
        hog_features[i*shape_features:(i+1)*shape_features] = HOG_features[i]
    
    # Essayer de voir si ça change les perfs qu'on renormalise à la fin avec toutes les features ensemble
    # pour envoyer un vecteur unitaire dans le truc... À voir
    
    return hog_features

def draw_line(y1, x1, y2, x2, size):
    from skimage.draw import line
    rr, cc = line(y1, x1, y2, x2)
    # On garde que les pixels dans l'image
    mask = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
    return rr[mask], cc[mask]

#%% 

# HOG_features = HOG(pict_test)
# time = timer(HOG(pict_test))

# debut = time.time()
# HOG_features = HOG(pict_test)
# fin = time.time()
# print(f"Temps d'exécution avec Numba : {fin - debut:.4f}s")

#%% Comparaison avec le descripteur de skimage 

from skimage.feature import hog

features, hog_image = hog(
    pict_test,
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    block_norm='L2',
    visualize=True,
    transform_sqrt=True,
    feature_vector=True
)

print("Taille du descripteur :", features.shape)

HOG_features = HOG(pict_test)

#%% 

with open('BD/cifar-10-batches-py/data_batch_1', 'rb') as f:
    train_cifar = pickle.load(f, encoding='bytes')
    
# Mise sous forme de liste
train_cifar = list(train_cifar.values())[2]