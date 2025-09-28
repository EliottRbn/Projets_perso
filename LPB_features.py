# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 23:57:05 2025

@author: eliot
"""

import cv2 
import numpy as np 
import os 
import numba 

#%% 

files = [file for file in os.listdir('BD/train') if file == "neutral"]
pictures = os.listdir(f"BD/train/{files[0]}")
pict_test = cv2.imread(f"BD/train/{files[0]}/{pictures[0]}", 0)
pict_test = cv2.resize(pict_test, dsize = (64,64)).astype("float64")

cv2.imshow('test',pict_test.astype("uint8"))

#%% Histogramme pour LPB

@numba.njit
def hist_LPB(Img_grid, n, nbins = 256):
    """
    Parameters
    ----------
    Img_grid : Array
        Grille de l'image obtenue par les descripteurs LPB
    n : Int
        Taille de la grille 
    nbins : Int, optional
       Représente les nombres binaires en base 8 contenant l'ensemble des nombres pouvant résulter des caractéristiques LPB.
       The default is 256.
       
    Returns
    -------
    hist : Array
        Histogramme de la grille 

    """
    hist = np.zeros(nbins, dtype= float)
    for i in range(n):
        for j in range(n):
            val = Img_grid[i,j]
            hist[val] += 1 

    return hist 

#%% Descripteurs LPB 

n,m = np.shape(pict_test)

def padded_img(img, pad):
    """
    Parameters
    ----------
    img : Array
        Image sur laquelle on souhaite appliquer le padding
    pad : Int
        "Amplitude" du padding

    Returns
    -------
    padded_image : Array
        Image avec padding appliqué

    """
    n,m = img.shape
    
    img = img.astype("uint8")
     
    padded_image = np.zeros((n + 2*pad, m + 2*pad), dtype = "uint8")
    
    # Remplissage de l'image au centre de la matrice de padding
    
    padded_image[1:-1,1:-1] = img
    
    return padded_image 

# Convertion en uint8 pour une optimisation de la mémoire et que ça soit plus optimisé
pict = padded_img(img = pict_test, pad = 1)

# Pour les conditions au bord, on ajoute juste un padding de zéro autour de l'image pour éviter d'introduire de l'information
LBP_img = np.zeros((n,m), dtype = "uint8")

# On suppose que les fenêtres utilisées sont carrées
win_size = 3 

# Python ne me laisse pas nommer la variable 2k pour 2^k, va savoir
k2 = np.array([2**k for k in range(8)], dtype = "uint8")

for i in range(n):
    for j in range(m):
        # Partie assez générale pour une fenêtre de taille quelconque 
        window = pict[(i + 1 - win_size//2) : (i + win_size//2 + 2) , (j + 1 - win_size//2) : (j + win_size//2 + 2)]
        threshold = window[win_size//2,win_size//2]
        window_vect = np.zeros(win_size*win_size - 1)
        
        # Partie moins générale et moins propre, à retravailler quand on aura le temps à y consacrer
        window_vect[:3] = window[0,:]
        window_vect[3] = window[1,2]
        window_vect[4:7] = window[2,::-1]
        window_vect[7] = window[1,0]
        
        # Convertion en binaire
        window_vect[window_vect < threshold] = 0
        window_vect[window_vect >= threshold] = 1
        
        # Calcul de la caractéristique LPB du pixel (i,j)
        LBP_img[i,j] = k2@window_vect
        
#%% Visualisation de l'image résultante 

cv2.imshow("Test LPB", LBP_img) # ça semble good 

#%% Histogrammes 

# Partition de l'image en grilles de taille 16x16 

grid_length = 16 

nb_grid = n // grid_length

hist_img = [] # Liste vide qui contiendra tous les histogrammes concaténés

for x in range(nb_grid):
    for y in range(nb_grid):
        Img_grid = LBP_img[x*grid_length : (x+1)*grid_length, y*grid_length : (y+1)*grid_length]
        hist_img = np.concatenate((hist_img, hist_LPB(Img_grid, grid_length)), axis = 0)
