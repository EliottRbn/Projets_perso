#%% Descripteurs Haar

import numba 
import numpy as np
import os
import cv2
import time 

#%% 

files = [file for file in os.listdir('BD/train') if file == "neutral"]
pictures = os.listdir(f"BD/train/{files[0]}")
pict_test = cv2.imread(f"BD/train/{files[0]}/{pictures[0]}", 0)
pict_test = cv2.resize(pict_test, dsize = (64,64)).astype("float64")

#%% 

### Fonctions pour les images intégrées et les images intégrées au carré ###

@numba.njit
def ii(img):
    """
    Parameters
    ----------
    img : Array
        Image en teinte de noir et blanc sur laquelle on calcule son image intégrale en parcourant les lignes et colonnes

    Returns
    -------
    I : Array
       Image intégrale de l'image initiale

    """
    # Taille de l'image 
    m,n = img.shape
    
    # Matrice qui sera l'image intégrale de l'image avec zero-padding pour l'initialisation
    I = np.zeros((m+1,n+1), dtype = "float64")
    
    # Boucle sur les lignes
    for i in range(1,m+1):
        # Accumulation sur les colonnes (à ligne fixe)
        s = np.zeros((1,n+1), dtype = "float64")
        
        # Boucle sur les colonnes
        for j in range(1,n+1):
            # Relation de réccurence (trouvée dans l'article Rapid Object Detection using a Boosted Cascade of Simple Features) 
            # pour le calcul au pixel i,j 
            s[0,j] = img[i-1,j-1] + s[0,j-1]
            I[i,j] = I[i-1,j] + s[0,j]
    
    return I[1:,1:]

@numba.njit
def sii(img):
    """
    Parameters
    ----------
    img : Array
        Image en teinte de noir et blanc sur laquelle on calcule son image intégrale au carré en parcourant les lignes et colonnes

    Returns
    -------
    I : Array
       Image intégrale au carré de l'image initiale

    """
    # Taille de l'image 
    m,n = img.shape
    
    # Matrice qui sera l'image intégrale au carré de l'image avec zero-padding 
    I = np.zeros((m+1,n+1), dtype = "float64")
    
    # Boucle sur les lignes
    for i in range(1,m+1):
        # Accumulation sur les colonnes (à ligne fixe)
        s = np.zeros((1,n+1), dtype = "float64")
    
        # Boucle sur les colonnes
        for j in range(1,n+1):
            # Relation de réccurence (trouvée dans l'article Rapid Object Detection using a Boosted Cascade of Simple Features) 
            # pour le calcul au pixel i,j 
            s[0,j] = img[i-1,j-1]**2 + s[0,j-1]
            I[i,j] = I[i-1,j] + s[0,j]
            
    return I[1:,1:]

#%% Fonction qui calcule les caractéristiques de Haar sur des images 64x64 

@numba.njit
def Haar_features(I):
    """
    Parameters
    ----------
    I : Array 
        Image dont on cherche à déterminer les features de Haar 

    Returns
    -------
    Features : Array
        Vecteur des features (normalisé) déterminé avec les filtres de Haar
    """
    # Vecteur nul qui contiendra les features de l'image 
    Features = np.zeros(928, dtype= "float64")

    # Récupération des images intégrales et images intégrales carrées
    
    Ii = ii(I)
    Isi = sii(I)
    
    # Nombre de boucles sur les lignes et colonnes pour les sous-fenêtres de taille 24x24 et 12x12
    
    k_24 = 6
    k_12 = 14    
    
    # Indice pour remplir les features 
    
    l = 0
    
    # Nombre de pixels/aire pour les différents filtres des fenêtres de taille 24x24 
    
    Area_rhv_24 = 24*12 # Aire pour les rectangles verticaux et horizontaux 
    Area_triple_24 = 24*8 # Aire pour les triples rectangles 
    Area_quadruple_24 = 12*12 # Aire pour les quadruples rectangles
    
    # Nombre de pixels/aire pour les différents filtres des fenêtres de taille 12x12
    
    Area_rhv_12 = 12*6 # Aire pour les rectangles verticaux et horizontaux 
    Area_triple_12 = 12*4 # Aire pour les triples rectangles 
    Area_quadruple_12 = 6*6 # Aire pour les quadruples rectangles
    
    # Tracking de la coordonnée x du point en haut à gauche de la fenêtre 
    pos_x = 0
    # Boucle sur les lignes des sous-fenêtres 24x24 
    for i in range(k_24):
        # Tracking de la coordonnée y du point en haut à gauche de la fenêtre
        pos_y = 0
        # Boucle sur les colonnes des sous-fenêtres 24x24
        for j in range(k_24):
            # Calcul des features pour les fenêtres rectangulaires verticales 
            mu_white = (1/Area_rhv_24)*(Ii[pos_x,pos_y] + Ii[pos_x + 23,pos_y + 11] - Ii[pos_x,pos_y + 11] - Ii[pos_x+23,pos_y])
            mu_black = (1/Area_rhv_24)*(Ii[pos_x,pos_y + 11] + Ii[pos_x + 23,pos_y + 23] - Ii[pos_x,pos_y + 23] - Ii[pos_x + 23, pos_y + 11])
            E_white = (1/Area_rhv_24)*(Isi[pos_x,pos_y] + Isi[pos_x + 23,pos_y + 11] - Isi[pos_x,pos_y + 11] - Isi[pos_x+23,pos_y])
            E_black = (1/Area_rhv_24)*(Isi[pos_x,pos_y + 11] + Isi[pos_x + 23,pos_y + 23] - Isi[pos_x,pos_y + 23] - Isi[pos_x + 23, pos_y + 11])
            
            sig_white = E_white - mu_white**2 
            sig_black = E_black - mu_black**2
            Features[l] = sig_black - sig_white # Convention pour le calcul des features : partie noire - partie blanche 

            # Calcul des features pour les fenêtres rectangulaires horizontales 
            
            mu_white = (1/Area_rhv_24)*(Ii[pos_x + 11,pos_y] + Ii[pos_x + 23,pos_y + 23] - Ii[pos_x + 11,pos_y + 23] - Ii[pos_x + 23,pos_y])
            mu_black = (1/Area_rhv_24)*(Ii[pos_x,pos_y] + Ii[pos_x + 11,pos_y + 23] - Ii[pos_x,pos_y + 23] - Ii[pos_x + 11, pos_y])
            E_white = (1/Area_rhv_24)*(Isi[pos_x + 11,pos_y] + Isi[pos_x + 23,pos_y + 23] - Isi[pos_x + 11,pos_y + 23] - Isi[pos_x + 23,pos_y])
            E_black = (1/Area_rhv_24)*(Isi[pos_x,pos_y] + Isi[pos_x + 11,pos_y + 23] - Isi[pos_x,pos_y + 23] - Isi[pos_x + 11, pos_y])
            
            sig_white = E_white - mu_white**2 
            sig_black = E_black - mu_black**2
            Features[l + 1] = sig_black - sig_white 
            
            # Calcul des features pour les fenêtres tri-rectangulaires  
            
            mu_white_1 = (1/Area_triple_24)*(Ii[pos_x,pos_y] + Ii[pos_x + 23, pos_y + 7] - Ii[pos_x,pos_y + 7] - Ii[pos_x+23,pos_y]) 
            mu_white_2 = (1/Area_triple_24)*(Ii[pos_x,pos_y+15] + Ii[pos_x + 23, pos_y + 23] - Ii[pos_x,pos_y + 23] - Ii[pos_x+23,pos_y + 15])
            mu_black = (1/Area_triple_24)*(Ii[pos_x,pos_y+7] + Ii[pos_x + 23, pos_y + 15] - Ii[pos_x,pos_y + 15] - Ii[pos_x+23,pos_y + 7])
            E_white_1 = (1/Area_triple_24)*(Isi[pos_x,pos_y] + Isi[pos_x + 23, pos_y + 7] - Isi[pos_x,pos_y + 7] - Isi[pos_x+23,pos_y]) 
            E_white_2 = (1/Area_triple_24)*(Isi[pos_x,pos_y+15] + Isi[pos_x + 23, pos_y + 23] - Isi[pos_x,pos_y + 23] - Isi[pos_x+23,pos_y + 15])
            E_black = (1/Area_triple_24)*(Isi[pos_x,pos_y+7] + Isi[pos_x + 23, pos_y + 15] - Isi[pos_x,pos_y + 15] - Isi[pos_x+23,pos_y + 7])
            
            sig_white_1 = E_white_1 - mu_white_1**2 
            sig_white_2 = E_white_2 - mu_white_2**2
            sig_black = E_black - mu_black**2
            Features[l + 2] = sig_black - sig_white_1 - sig_white_2            
            
            # Calcul des features pour les fenêtres quadri-rectangulaires  
            
            mu_white_1 = (1/Area_quadruple_24)*(Ii[pos_x,pos_y] + Ii[pos_x + 11,pos_y + 11] - Ii[pos_x,pos_y + 11] - Ii[pos_x + 11,pos_y])
            mu_white_2 = (1/Area_quadruple_24)*(Ii[pos_x+11,pos_y+11] + Ii[pos_x+23,pos_y+23] - Ii[pos_x+11,pos_y+23] - Ii[pos_x+23,pos_y + 11])
            mu_black_1 = (1/Area_quadruple_24)*(Ii[pos_x,pos_y+11] + Ii[pos_x+11,pos_y+23] - Ii[pos_x,pos_y+23] - Ii[pos_x+11,pos_y+11])
            mu_black_2 = (1/Area_quadruple_24)*(Ii[pos_x+11,pos_y] + Ii[pos_x+23,pos_y+11] - Ii[pos_x+11,pos_y+11] - Ii[pos_x+23,pos_y])
            E_white_1 = (1/Area_quadruple_24)*(Isi[pos_x,pos_y] + Isi[pos_x + 11,pos_y + 11] - Isi[pos_x,pos_y + 11] - Isi[pos_x + 11,pos_y])
            E_white_2 = (1/Area_quadruple_24)*(Isi[pos_x+11,pos_y+11] + Isi[pos_x+23,pos_y+23] - Isi[pos_x+11,pos_y+23] - Isi[pos_x+23,pos_y + 11])
            E_black_1 = (1/Area_quadruple_24)*(Isi[pos_x,pos_y+11] + Isi[pos_x+11,pos_y+23] - Isi[pos_x,pos_y+23] - Isi[pos_x+11,pos_y+11])
            E_black_2 = (1/Area_quadruple_24)*(Isi[pos_x+11,pos_y] + Isi[pos_x+23,pos_y+11] - Isi[pos_x+11,pos_y+11] - Isi[pos_x+23,pos_y])
            
            sig_white_1 = E_white_1 - mu_white_1**2 
            sig_white_2 = E_white_2 - mu_white_2**2 
            sig_black_1 = E_black_1 - mu_black_1**2 
            sig_black_2 = E_black_2 - mu_black_2**2
            Features[l + 3] = sig_black_1 + sig_black_2 - sig_white_1 - sig_white_2
            
            pos_y += 8 
            l += 4
        pos_x += 8 
        
    # Tracking de la coordonnée x du point en haut à gauche de la fenêtre 
    pos_x = 0
    # Boucle sur les lignes des sous-fenêtres 12x12 
    for i in range(k_12):
        # Tracking de la coordonnée y du point en haut à gauche de la fenêtre
        pos_y = 0
        # Boucle sur les colonnes des sous-fenêtres 12x12
        for j in range(k_12):
            # Calcul des features pour les fenêtres rectangulaires verticales 
            mu_white = (1/Area_rhv_12)*(Ii[pos_x,pos_y] + Ii[pos_x + 11,pos_y + 5] - Ii[pos_x,pos_y + 5] - Ii[pos_x+11,pos_y])
            mu_black = (1/Area_rhv_12)*(Ii[pos_x,pos_y + 5] + Ii[pos_x + 11,pos_y + 11] - Ii[pos_x,pos_y + 11] - Ii[pos_x + 11, pos_y + 5])
            E_white = (1/Area_rhv_12)*(Isi[pos_x,pos_y] + Isi[pos_x + 11,pos_y + 5] - Isi[pos_x,pos_y + 5] - Isi[pos_x+11,pos_y])
            E_black = (1/Area_rhv_12)*(Isi[pos_x,pos_y + 5] + Isi[pos_x + 11,pos_y + 11] - Isi[pos_x,pos_y + 11] - Isi[pos_x + 11, pos_y + 5])
            
            sig_white = E_white - mu_white**2 
            sig_black = E_black - mu_black**2
            Features[l] = sig_black - sig_white # Convention pour le calcul des features : partie noire - partie blanche 

            # Calcul des features pour les fenêtres rectangulaires horizontales 
            
            mu_white = (1/Area_rhv_12)*(Ii[pos_x + 5,pos_y] + Ii[pos_x + 11,pos_y + 11] - Ii[pos_x + 5,pos_y + 11] - Ii[pos_x + 11,pos_y])
            mu_black = (1/Area_rhv_12)*(Ii[pos_x,pos_y] + Ii[pos_x + 5,pos_y + 11] - Ii[pos_x,pos_y + 11] - Ii[pos_x + 5, pos_y])
            E_white = (1/Area_rhv_12)*(Isi[pos_x + 5,pos_y] + Isi[pos_x + 11,pos_y + 11] - Isi[pos_x + 5,pos_y + 11] - Isi[pos_x + 11,pos_y])
            E_black = (1/Area_rhv_12)*(Isi[pos_x,pos_y] + Isi[pos_x + 5,pos_y + 11] - Isi[pos_x,pos_y + 11] - Isi[pos_x + 5, pos_y])
            
            sig_white = E_white - mu_white**2 
            sig_black = E_black - mu_black**2
            Features[l + 1] = sig_black - sig_white 
            
            # Calcul des features pour les fenêtres tri-rectangulaires  
            
            mu_white_1 = (1/Area_triple_12)*(Ii[pos_x,pos_y] + Ii[pos_x + 11, pos_y + 4] - Ii[pos_x,pos_y + 4] - Ii[pos_x+11,pos_y]) 
            mu_white_2 = (1/Area_triple_12)*(Ii[pos_x,pos_y+7] + Ii[pos_x + 11, pos_y + 11] - Ii[pos_x,pos_y + 11] - Ii[pos_x+11,pos_y + 7])
            mu_black = (1/Area_triple_12)*(Ii[pos_x,pos_y+4] + Ii[pos_x + 11, pos_y + 7] - Ii[pos_x,pos_y + 7] - Ii[pos_x+11,pos_y + 4])
            E_white_1 = (1/Area_triple_12)*(Isi[pos_x,pos_y] + Isi[pos_x + 11, pos_y + 4] - Isi[pos_x,pos_y + 4] - Isi[pos_x+11,pos_y]) 
            E_white_2 = (1/Area_triple_12)*(Isi[pos_x,pos_y+7] + Isi[pos_x + 11, pos_y + 11] - Isi[pos_x,pos_y + 11] - Isi[pos_x+11,pos_y + 7])
            E_black = (1/Area_triple_12)*(Isi[pos_x,pos_y+4] + Isi[pos_x + 11, pos_y + 7] - Isi[pos_x,pos_y + 7] - Isi[pos_x+11,pos_y + 4])
            
            sig_white_1 = E_white_1 - mu_white_1**2 
            sig_white_2 = E_white_2 - mu_white_2**2
            sig_black = E_black - mu_black**2
            Features[l + 2] = sig_black - sig_white_1 - sig_white_2            
            
            # Calcul des features pour les fenêtres quadri-rectangulaires  
            
            mu_white_1 = (1/Area_quadruple_12)*(Ii[pos_x,pos_y] + Ii[pos_x + 5,pos_y + 5] - Ii[pos_x,pos_y + 5] - Ii[pos_x + 5,pos_y])
            mu_white_2 = (1/Area_quadruple_12)*(Ii[pos_x+5,pos_y+5] + Ii[pos_x+11,pos_y+11] - Ii[pos_x+5,pos_y+11] - Ii[pos_x+11,pos_y + 5])
            mu_black_1 = (1/Area_quadruple_12)*(Ii[pos_x,pos_y+5] + Ii[pos_x+5,pos_y+11] - Ii[pos_x,pos_y+11] - Ii[pos_x+5,pos_y+5])
            mu_black_2 = (1/Area_quadruple_12)*(Ii[pos_x+5,pos_y] + Ii[pos_x+11,pos_y+5] - Ii[pos_x+5,pos_y+5] - Ii[pos_x+11,pos_y])
            E_white_1 = (1/Area_quadruple_12)*(Isi[pos_x,pos_y] + Isi[pos_x + 5,pos_y + 5] - Isi[pos_x,pos_y + 5] - Isi[pos_x + 5,pos_y])
            E_white_2 = (1/Area_quadruple_12)*(Isi[pos_x+5,pos_y+5] + Isi[pos_x+11,pos_y+11] - Isi[pos_x+5,pos_y+11] - Isi[pos_x+11,pos_y + 5])
            E_black_1 = (1/Area_quadruple_12)*(Isi[pos_x,pos_y+5] + Isi[pos_x+5,pos_y+11] - Isi[pos_x,pos_y+11] - Isi[pos_x+5,pos_y+5])
            E_black_2 = (1/Area_quadruple_12)*(Isi[pos_x+5,pos_y] + Isi[pos_x+11,pos_y+5] - Isi[pos_x+5,pos_y+5] - Isi[pos_x+11,pos_y])
            
            sig_white_1 = E_white_1 - mu_white_1**2 
            sig_white_2 = E_white_2 - mu_white_2**2 
            sig_black_1 = E_black_1 - mu_black_1**2 
            sig_black_2 = E_black_2 - mu_black_2**2
            Features[l + 3] = sig_black_1 + sig_black_2 - sig_white_1 - sig_white_2
            
            pos_y += 4
            l += 4
        pos_x += 4
    
    # Normalisation des features 
    norm = np.sqrt(np.sum(Features ** 2))
    if norm != 0.0:
        Features = Features / norm
    
    return Features 

#%% 

debut = time.time()
features = Haar_features(pict_test)
print(f"Temps d'exécution du code sur une image 64x64 : {time.time() - debut:.10f}")

#%% 

A = np.array([[2, 4, 3, 4],
              [3, 7, 1, 6],
              [1, 2, 4, 7],
              [3, 4, 6, 7]])
i_A = ii(pict_test)
si_A = sii(pict_test)







