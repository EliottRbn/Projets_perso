# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 21:33:48 2024

@author: eliot
"""

import cv2
import numpy as np
import numba 
import matplotlib.pyplot as plt 
import SVM_class as SVMC
import pickle
import os
import time

#%% Variables globales 

global sig
global c 
global d

sig = 0.9
c = 3 
d = 2

# Retravailler la normalisation des images !

#%% Fonction à appeler pour tester la rapidité d'une fonction (marche que si c'est non jité)

def timer(funct):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = funct(*args, **kwargs)
        end = time.time()
        print(f"Temps d'exécution : {end - start:.4f}s")
        return result
    return wrapper

#%% Routine pour choisir des fichiers tests 

def raw_data(u,v):
    nb_u = len(u)
    
    if v is not None:
        nb_v = len(v)
        
        X = np.empty((nb_u + nb_v,4096), dtype = "float64")
        for (i,img_u),(j,img_v) in zip(enumerate(u),enumerate(v)):
            X[i,:] = u[i].flatten(order = 'C') / np.linalg.norm(u[i].flatten(order = 'C'))
            X[nb_u+j,:] = v[j].flatten(order = 'C') / np.linalg.norm(v[j].flatten(order = 'C'))
        
        Y = np.zeros(nb_u + nb_v)
        Y[:nb_u] = 1
        Y[nb_u:] = -1 
        
        return X,Y   
    
    # Gestion pour la caméra 
    elif v is None:

        X = u.flatten(order = 'C') / np.linalg.norm(u.flatten(order = 'C'))
        
        return X

def Haar_data(u,v):
    nb_u = len(u)
    
    if v is not None:
        nb_v = len(v)
        
        # Features de taille fixe donc on initialise à la bonne taille 
        X = np.empty((nb_u + nb_v,928), dtype= "float64")
        
        for (i,img_u),(j,img_v) in zip(enumerate(u),enumerate(v)):
            X[i,:] = Haar_features(img_u)
            X[nb_u + j,:] = Haar_features(img_v)
        
        Y = np.zeros(nb_u + nb_v)
        Y[:nb_u] = 1 
        Y[nb_u:] = -1 
        
        return X,Y
    
    # Gestion pour la caméra
    elif v is None:
        X = Haar_features(u)
        
        return X
    

def HOG_data(u,v):
    nb_u = len(u)
    
    if v is not None:
        nb_v = len(v)
        
        # Les features sont de taille fixes donc on peut juste initialiser à la bonne taille directement
        X = np.zeros((nb_u + nb_v,1764), dtype = "float64")
        
        for (i,img_u),(j,img_v) in zip(enumerate(u),enumerate(v)):
            X[i,:] = HOG(img_u)
            X[nb_u + j,:] = HOG(img_v)
        
        Y = np.zeros(nb_u + nb_v)
        Y[:nb_u] = 1 
        Y[nb_u:] = - 1
        
        return X,Y
    
    # Gestion pour la caméra
    elif v is None:
        X = HOG(u)
        
        return X

def LBP_data(u,v):
    nb_u = len(u)
    
    if v is not None:
        nb_v = len(v)
        
        # Les features sont de tailles 4096 donc on peut juste initialiser à la bonne taille directement 
        X = np.zeros((nb_u + nb_v, 4096), dtype = 'float64')
        
        for (i,img_u), (j,img_v) in zip(enumerate(u),enumerate(v)):
            X[i,:] = LBP(img_u)
            X[nb_u + j,:] = LBP(img_v)
        
        Y = np.zeros(nb_u + nb_v)
        Y[:nb_u] = 1
        Y[nb_u:] = -1 
        
        return X,Y
    
    # Gestion pour la caméra
    elif v is None:
        X = LBP(u)
        
        return X

def HOG_Haar_data(u,v):
    nb_u = len(u)
    
    if v is not None:
        nb_v = len(v)
    
        # Les features sont de taille fixe donc on initialise comme ça
        X = np.empty((nb_u + nb_v,928+1764), dtype = "float64")
        
        for (i,img_u),(j,img_v) in zip(enumerate(u),enumerate(v)):
            X[i,:928] = Haar_features(img_u)
            X[nb_u + j,:928] = Haar_features(img_v)
            X[i, 928:] = HOG(img_u)
            X[nb_u + j,928:] = HOG(img_v)
    
        Y = np.zeros(nb_u + nb_v)
        Y[:nb_u] = 1 
        Y[nb_u:] = -1 
               
        # Essayer de voir si on a un intérêtà tout renormaliser après ou si c'est ok comme ça niveau perfo
    
        return X,Y
    
    elif v is None: # Pour l'affichage à la caméra, donc une seule image on s'en bat les couilles
        X = np.empty((1,928+1764), dtype = "float64")
        X[0,:928] = Haar_features(u)
        X[0,928:] = HOG(u)
        
        return X

def HOG_LBP_data(u,v):
    nb_u = len(u)
    
    if v is not None:
        nb_v = len(v)
    
        # Les features sont de taille fixe donc on initialise comme ça
        X = np.empty((nb_u + nb_v,1764+4096), dtype = "float64")
        
        for (i,img_u),(j,img_v) in zip(enumerate(u),enumerate(v)):
            X[i,:1764] = HOG(img_u)
            X[nb_u + j,:1764] = HOG(img_v)
            X[i, 1764:] = LBP(img_u)
            X[nb_u + j,1764:] = LBP(img_v)
    
        Y = np.zeros(nb_u + nb_v)
        Y[:nb_u] = 1 
        Y[nb_u:] = -1 
    
        return X,Y
    
    elif v is None: # Pour l'affichage à la caméra, donc une seule image on s'en bat les couilles
        X = np.empty((1,1764+4096), dtype = "float64")
        X[0,:1764] = HOG(u)
        X[0,1764:] = LBP(u)
        
        return X

def Haar_LBP_data(u,v):
    nb_u = len(u)
    
    if v is not None:
        nb_v = len(v)
    
        # Les features sont de taille fixe donc on initialise comme ça
        X = np.empty((nb_u + nb_v,928+4096), dtype = "float64")
        
        for (i,img_u),(j,img_v) in zip(enumerate(u),enumerate(v)):
            X[i,:928] = Haar_features(img_u)
            X[nb_u + j,:928] = Haar_features(img_v)
            X[i, 928:] = LBP(img_u)
            X[nb_u + j,928:] = LBP(img_v)
    
        Y = np.zeros(nb_u + nb_v)
        Y[:nb_u] = 1 
        Y[nb_u:] = -1 
    
        return X,Y
    
    elif v is None: # Pour l'affichage à la caméra, donc une seule image on s'en bat les couilles
        X = np.empty((1,928+4096), dtype = "float64")
        X[0,:928] = Haar_features(u)
        X[0,928:] = LBP(u)
        
        return X
    
def Haar_HOG_LBP_data(u,v):
    nb_u = len(u)
    
    if v is not None:
        nb_v = len(v)
    
        # Les features sont de taille fixe donc on initialise comme ça
        X = np.empty((nb_u + nb_v,928+1764+4096), dtype = "float64")
        
        for (i,img_u),(j,img_v) in zip(enumerate(u),enumerate(v)):
            X[i,:928] = Haar_features(img_u)
            X[nb_u + j,:928] = Haar_features(img_v)
            X[i, 928:928+1764] = HOG(img_u)
            X[nb_u + j,928:928+1764] = HOG(img_v)
            X[i, 928+1764:] = LBP(img_u)
            X[nb_u + j,928+1764:] = LBP(img_v)
    
        Y = np.zeros(nb_u + nb_v)
        Y[:nb_u] = 1 
        Y[nb_u:] = -1 
    
        return X,Y
    
    elif v is None: # Pour l'affichage à la caméra, donc une seule image on s'en bat les couilles
        X = np.empty((1,928+1764+4096), dtype = "float64")
        X[0,:928] = Haar_features(u)
        X[0,928:928+1764] = HOG(u)
        X[0,928+1764:] = LBP(u)
        
        return X

#%% 

def preprocess_image_cam(img, target_size=(64, 64)):
    """
    Parameters
    ----------
    img : Array
        Image d'entrée non carrée (e.g. de taille 720x480) à padder 
    target_size : Tuple 
        Taille de l'image finale une fois le padding fait puis après avoir appliqué le reshape. The default is (64, 64).

    Returns
    -------
    Array
        Image reshape après padding pour conserver les formes 

    """
    # On récupère la shape de notre image 
    h, w = img.shape[:2]
    
    # Centrage dans une image carrée
    max_dim = max(h, w)
    padded = np.zeros((max_dim, max_dim), dtype=img.dtype)
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    
    # Rajoute l'image dans cette image carrée noir 
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img
    
    # Return le resize mais sur un carré plutôt que sur un rectangle pour conserver les formes relatives de l'image
    return cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)

#%% 

def add_padding(img, target_size = (64,64), value_padding = 128):
    # Pour des images déjà carrée mais où l'on ajoute volontairement du padding pour habituer l'algo à en avoir dans les données
    # et donc ne pas considérer le padding comme une information discriminante pour les images 
    # Fonction faite sur mesure pour la BDD avec des images de taille 32x32 et 48x48, à modifier si on veut plus de généralité
    # Taille de l'image 
    n,m = img.shape
    if n == 32:
        pad = 4
    elif n == 48:
        pad = 2
    else:
        raise ValueError
        print("Les images ne correspondent pas aux données attendues.")
        
    padded = np.ones((n + 2*pad, m + 2*pad), dtype = img.dtype)*value_padding 
    padded[pad:(n+pad),pad:(m+pad)] = img 
    
    return cv2.resize(padded, target_size, interpolation = cv2.INTER_AREA) 

#%%

def routine_test_padded(path_u,path_v,nb_u,nb_v, config = raw_data, percentage = 0.3):
    nb_padded_u = int(nb_u*percentage) 
    nb_padded_v = int(nb_v*percentage)
    
    # Chargement des données de v
    with open(path_v, 'rb') as f:
        train_cifar = pickle.load(f, encoding='bytes')
        
    # Mise sous forme de liste
    train_cifar = list(train_cifar.values())[2]

    # Chargement du dossier vers les images de u
    files_u = os.listdir(path_u)
    
    # Vecteur de u contenant les images en teinte de gris
    u = []
    
    # Remplissage du vecteur u
    compt = 0
    compt_padded = 0
    compt_file = 0
    while compt < nb_u:
        sous_compteur = 0
        for file in files_u:
            pict = os.listdir(f"{path_u}/{file}")
            random_value = np.random.uniform()
            try:
                u_bw = cv2.imread(f"{path_u}/{file}/{pict[compt_file]}",0)
            except IndexError:
                files_u.remove(file) # On retire les dossiers trop petits qui n'ont plus d'images à fournir 

            if random_value >= 0.5 and compt_padded < nb_padded_u:
                u_bw = add_padding(u_bw).astype("float64")
                u.append(u_bw)
                compt_padded += 1
                sous_compteur += 1
            else:
                u.append(cv2.resize(u_bw,(64,64), interpolation = cv2.INTER_AREA).astype("float64"))
                sous_compteur += 1
            
            if compt + sous_compteur >= nb_u:
                break 
                
        compt += sous_compteur 
        compt_file += 1
                
    # print(f"{compt_padded} images positives ont subit du padding")
    # Liste de v contenant les images en teinte de gris 
    v = []
    
    compt_padded = 0
    # Remplissage du vecteur v
    for i in range(0,nb_v):
        img_v = np.zeros((32,32,3))
        for j in range(3):
            img_v[:,:,j] = train_cifar[i,(1024*j):(1024*(j+1))].reshape(32,32)
        img_v_bw = cv2.cvtColor(img_v.astype("uint8"), cv2.COLOR_BGR2GRAY)
        random_value = np.random.uniform()
        if random_value >= 0.5 and compt_padded < nb_padded_v:
            img_v_bw = add_padding(img_v_bw).astype("float64")
            v.append(img_v_bw)
            compt_padded += 1 
        else:
            v.append(cv2.resize(img_v_bw, (64,64), interpolation = cv2.INTER_AREA).astype("float64"))
    
    # print(f"{compt_padded} images négatives ont subit du padding")
    
    return config(u,v)

#%%

def routine_test(path_u,path_v,nb_u,nb_v, config = raw_data):
    # Chargement des données de v
    with open(path_v, 'rb') as f:
        train_cifar = pickle.load(f, encoding='bytes')
        
    # Mise sous forme de liste
    train_cifar = list(train_cifar.values())[2]

    # Chargement du dossier vers les images de u
    files_u = os.listdir(path_u)
    
    # Vecteur de u contenant les images en teinte de gris
    u = []
    
    # Remplissage du vecteur u
    compt = 0
    for file in files_u:
        if file == "neutral":
            pict = os.listdir(f"{path_u}/{file}")
            for picture in pict:
                u_bw = cv2.imread(f"{path_u}/{file}/{picture}",0)
                u.append(cv2.resize(u_bw,(64,64), interpolation = cv2.INTER_AREA).astype("float64"))
                compt += 1 
                if compt == nb_u :
                    break

    # Liste de v contenant les images en teinte de gris 
    v = []
    
    # Remplissage du vecteur v
    for i in range(0,nb_v):
        img_v = np.zeros((32,32,3))
        for j in range(3):
            img_v[:,:,j] = train_cifar[i,(1024*j):(1024*(j+1))].reshape(32,32)
        img_v_bw = cv2.cvtColor(img_v.astype("uint8"), cv2.COLOR_BGR2GRAY).astype("float64")
        v.append(cv2.resize(img_v_bw, (64,64), interpolation = cv2.INTER_AREA))
    
    return config(u,v)
   
#%% Descripteurs HOG : Fait pour des images de taille 64x64 mais être facilement adapté juste en changeant la taille des cellules

@numba.njit
def hist_OG(G, theta, n, nbins = 9, min_val = 0, max_val = 180):
    hist = np.zeros(nbins, dtype='float64')
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
    cell_storage = np.empty((nbins*fraction_size,fraction_size), dtype = 'float64')
    
    for j in range(fraction_size):
        for i in range(fraction_size):
            window_G = G[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size]
            window_theta = theta[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size]
            cell_storage[i*nbins:(i+1)*nbins,j] = hist_OG(window_G,window_theta, cell_size)
    
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
    
    hog_features = np.zeros(nb_features*shape_features, dtype = 'float64')
    
    for i,feature in enumerate(HOG_features):
        hog_features[i*shape_features:(i+1)*shape_features] = HOG_features[i]
    
    # Essayer de voir si ça change les perfs qu'on renormalise à la fin avec toutes les features ensemble
    # pour envoyer un vecteur unitaire dans le truc... À voir
    
    return hog_features

#%% Descripteurs Haar

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
def rect_ver_24(Area,Ii,Isi,pos_x,pos_y):
    # Calcul des features pour les fenêtres rectangulaires verticales 
    
    mu_white = (1/Area)*(Ii[pos_x,pos_y] + Ii[pos_x + 23,pos_y + 11] - Ii[pos_x,pos_y + 11] - Ii[pos_x+23,pos_y])
    mu_black = (1/Area)*(Ii[pos_x,pos_y + 11] + Ii[pos_x + 23,pos_y + 23] - Ii[pos_x,pos_y + 23] - Ii[pos_x + 23, pos_y + 11])
    E_white = (1/Area)*(Isi[pos_x,pos_y] + Isi[pos_x + 23,pos_y + 11] - Isi[pos_x,pos_y + 11] - Isi[pos_x+23,pos_y])
    E_black = (1/Area)*(Isi[pos_x,pos_y + 11] + Isi[pos_x + 23,pos_y + 23] - Isi[pos_x,pos_y + 23] - Isi[pos_x + 23, pos_y + 11])
    
    sig_white = E_white - mu_white**2 
    sig_black = E_black - mu_black**2
    
    return sig_black - sig_white # Convention pour le calcul des features : partie noire - partie blanche 

@numba.njit
def rect_hor_24(Area,Ii,Isi,pos_x,pos_y):
    # Calcul des features pour les fenêtres rectangulaires horizontales 
    
    mu_white = (1/Area)*(Ii[pos_x + 11,pos_y] + Ii[pos_x + 23,pos_y + 23] - Ii[pos_x + 11,pos_y + 23] - Ii[pos_x + 23,pos_y])
    mu_black = (1/Area)*(Ii[pos_x,pos_y] + Ii[pos_x + 11,pos_y + 23] - Ii[pos_x,pos_y + 23] - Ii[pos_x + 11, pos_y])
    E_white = (1/Area)*(Isi[pos_x + 11,pos_y] + Isi[pos_x + 23,pos_y + 23] - Isi[pos_x + 11,pos_y + 23] - Isi[pos_x + 23,pos_y])
    E_black = (1/Area)*(Isi[pos_x,pos_y] + Isi[pos_x + 11,pos_y + 23] - Isi[pos_x,pos_y + 23] - Isi[pos_x + 11, pos_y])
    
    sig_white = E_white - mu_white**2 
    sig_black = E_black - mu_black**2
    
    return sig_black - sig_white # Convention pour le calcul des features : partie noire - partie blanche 

@numba.njit
def rect_triple_24(Area, Ii, Isi, pos_x, pos_y):
    # Calcul des features pour les fenêtres tri-rectangulaires  
    
    mu_white_1 = (1/Area)*(Ii[pos_x,pos_y] + Ii[pos_x + 23, pos_y + 7] - Ii[pos_x,pos_y + 7] - Ii[pos_x+23,pos_y]) 
    mu_white_2 = (1/Area)*(Ii[pos_x,pos_y+15] + Ii[pos_x + 23, pos_y + 23] - Ii[pos_x,pos_y + 23] - Ii[pos_x+23,pos_y + 15])
    mu_black = (1/Area)*(Ii[pos_x,pos_y+7] + Ii[pos_x + 23, pos_y + 15] - Ii[pos_x,pos_y + 15] - Ii[pos_x+23,pos_y + 7])
    E_white_1 = (1/Area)*(Isi[pos_x,pos_y] + Isi[pos_x + 23, pos_y + 7] - Isi[pos_x,pos_y + 7] - Isi[pos_x+23,pos_y]) 
    E_white_2 = (1/Area)*(Isi[pos_x,pos_y+15] + Isi[pos_x + 23, pos_y + 23] - Isi[pos_x,pos_y + 23] - Isi[pos_x+23,pos_y + 15])
    E_black = (1/Area)*(Isi[pos_x,pos_y+7] + Isi[pos_x + 23, pos_y + 15] - Isi[pos_x,pos_y + 15] - Isi[pos_x+23,pos_y + 7])
    
    sig_white_1 = E_white_1 - mu_white_1**2 
    sig_white_2 = E_white_2 - mu_white_2**2
    sig_black = E_black - mu_black**2
    
    return sig_black - sig_white_1 - sig_white_2 # Convention pour le calcul des features : partie noire - partie blanche 
 
@numba.njit   
def rect_quadruple_24(Area, Ii, Isi, pos_x, pos_y):
    # Calcul des features pour les fenêtres quadri-rectangulaires
    
    mu_white_1 = (1/Area)*(Ii[pos_x,pos_y] + Ii[pos_x + 11,pos_y + 11] - Ii[pos_x,pos_y + 11] - Ii[pos_x + 11,pos_y])
    mu_white_2 = (1/Area)*(Ii[pos_x+11,pos_y+11] + Ii[pos_x+23,pos_y+23] - Ii[pos_x+11,pos_y+23] - Ii[pos_x+23,pos_y + 11])
    mu_black_1 = (1/Area)*(Ii[pos_x,pos_y+11] + Ii[pos_x+11,pos_y+23] - Ii[pos_x,pos_y+23] - Ii[pos_x+11,pos_y+11])
    mu_black_2 = (1/Area)*(Ii[pos_x+11,pos_y] + Ii[pos_x+23,pos_y+11] - Ii[pos_x+11,pos_y+11] - Ii[pos_x+23,pos_y])
    E_white_1 = (1/Area)*(Isi[pos_x,pos_y] + Isi[pos_x + 11,pos_y + 11] - Isi[pos_x,pos_y + 11] - Isi[pos_x + 11,pos_y])
    E_white_2 = (1/Area)*(Isi[pos_x+11,pos_y+11] + Isi[pos_x+23,pos_y+23] - Isi[pos_x+11,pos_y+23] - Isi[pos_x+23,pos_y + 11])
    E_black_1 = (1/Area)*(Isi[pos_x,pos_y+11] + Isi[pos_x+11,pos_y+23] - Isi[pos_x,pos_y+23] - Isi[pos_x+11,pos_y+11])
    E_black_2 = (1/Area)*(Isi[pos_x+11,pos_y] + Isi[pos_x+23,pos_y+11] - Isi[pos_x+11,pos_y+11] - Isi[pos_x+23,pos_y])
    
    sig_white_1 = E_white_1 - mu_white_1**2 
    sig_white_2 = E_white_2 - mu_white_2**2 
    sig_black_1 = E_black_1 - mu_black_1**2 
    sig_black_2 = E_black_2 - mu_black_2**2
    
    return sig_black_1 + sig_black_2 - sig_white_1 - sig_white_2 # Convention pour le calcul des features : partie noire - partie blanche

@numba.njit
def rect_ver_12(Area,Ii,Isi,pos_x,pos_y):
    # Calcul des features pour les fenêtres rectangulaires verticales 
    
    mu_white = (1/Area)*(Ii[pos_x,pos_y] + Ii[pos_x + 11,pos_y + 5] - Ii[pos_x,pos_y + 5] - Ii[pos_x+11,pos_y])
    mu_black = (1/Area)*(Ii[pos_x,pos_y + 5] + Ii[pos_x + 11,pos_y + 11] - Ii[pos_x,pos_y + 11] - Ii[pos_x + 11, pos_y + 5])
    E_white = (1/Area)*(Isi[pos_x,pos_y] + Isi[pos_x + 11,pos_y + 5] - Isi[pos_x,pos_y + 5] - Isi[pos_x+11,pos_y])
    E_black = (1/Area)*(Isi[pos_x,pos_y + 5] + Isi[pos_x + 11,pos_y + 11] - Isi[pos_x,pos_y + 11] - Isi[pos_x + 11, pos_y + 5])
    
    sig_white = E_white - mu_white**2 
    sig_black = E_black - mu_black**2
    
    return sig_black - sig_white # Convention pour le calcul des features : partie noire - partie blanche 

@numba.njit
def rect_hor_12(Area,Ii,Isi,pos_x,pos_y):
    # Calcul des features pour les fenêtres rectangulaires horizontales 
    
    mu_white = (1/Area)*(Ii[pos_x + 5,pos_y] + Ii[pos_x + 11,pos_y + 11] - Ii[pos_x + 5,pos_y + 11] - Ii[pos_x + 11,pos_y])
    mu_black = (1/Area)*(Ii[pos_x,pos_y] + Ii[pos_x + 5,pos_y + 11] - Ii[pos_x,pos_y + 11] - Ii[pos_x + 5, pos_y])
    E_white = (1/Area)*(Isi[pos_x + 5,pos_y] + Isi[pos_x + 11,pos_y + 11] - Isi[pos_x + 5,pos_y + 11] - Isi[pos_x + 11,pos_y])
    E_black = (1/Area)*(Isi[pos_x,pos_y] + Isi[pos_x + 5,pos_y + 11] - Isi[pos_x,pos_y + 11] - Isi[pos_x + 5, pos_y])
    
    sig_white = E_white - mu_white**2 
    sig_black = E_black - mu_black**2
    
    return sig_black - sig_white 

@numba.njit
def rect_triple_12(Area, Ii, Isi, pos_x, pos_y):
    # Calcul des features pour les fenêtres tri-rectangulaires  
    
    mu_white_1 = (1/Area)*(Ii[pos_x,pos_y] + Ii[pos_x + 11, pos_y + 4] - Ii[pos_x,pos_y + 4] - Ii[pos_x+11,pos_y]) 
    mu_white_2 = (1/Area)*(Ii[pos_x,pos_y+7] + Ii[pos_x + 11, pos_y + 11] - Ii[pos_x,pos_y + 11] - Ii[pos_x+11,pos_y + 7])
    mu_black = (1/Area)*(Ii[pos_x,pos_y+4] + Ii[pos_x + 11, pos_y + 7] - Ii[pos_x,pos_y + 7] - Ii[pos_x+11,pos_y + 4])
    E_white_1 = (1/Area)*(Isi[pos_x,pos_y] + Isi[pos_x + 11, pos_y + 4] - Isi[pos_x,pos_y + 4] - Isi[pos_x+11,pos_y]) 
    E_white_2 = (1/Area)*(Isi[pos_x,pos_y+7] + Isi[pos_x + 11, pos_y + 11] - Isi[pos_x,pos_y + 11] - Isi[pos_x+11,pos_y + 7])
    E_black = (1/Area)*(Isi[pos_x,pos_y+4] + Isi[pos_x + 11, pos_y + 7] - Isi[pos_x,pos_y + 7] - Isi[pos_x+11,pos_y + 4])
    
    sig_white_1 = E_white_1 - mu_white_1**2 
    sig_white_2 = E_white_2 - mu_white_2**2
    sig_black = E_black - mu_black**2   
    
    return sig_black - sig_white_1 - sig_white_2 # Convention pour le calcul des features : partie noire - partie blanche 
    
@numba.njit
def rect_quadruple_12(Area, Ii, Isi, pos_x, pos_y):
    # Calcul des features pour les fenêtres quadri-rectangulaires  
    
    mu_white_1 = (1/Area)*(Ii[pos_x,pos_y] + Ii[pos_x + 5,pos_y + 5] - Ii[pos_x,pos_y + 5] - Ii[pos_x + 5,pos_y])
    mu_white_2 = (1/Area)*(Ii[pos_x+5,pos_y+5] + Ii[pos_x+11,pos_y+11] - Ii[pos_x+5,pos_y+11] - Ii[pos_x+11,pos_y + 5])
    mu_black_1 = (1/Area)*(Ii[pos_x,pos_y+5] + Ii[pos_x+5,pos_y+11] - Ii[pos_x,pos_y+11] - Ii[pos_x+5,pos_y+5])
    mu_black_2 = (1/Area)*(Ii[pos_x+5,pos_y] + Ii[pos_x+11,pos_y+5] - Ii[pos_x+5,pos_y+5] - Ii[pos_x+11,pos_y])
    E_white_1 = (1/Area)*(Isi[pos_x,pos_y] + Isi[pos_x + 5,pos_y + 5] - Isi[pos_x,pos_y + 5] - Isi[pos_x + 5,pos_y])
    E_white_2 = (1/Area)*(Isi[pos_x+5,pos_y+5] + Isi[pos_x+11,pos_y+11] - Isi[pos_x+5,pos_y+11] - Isi[pos_x+11,pos_y + 5])
    E_black_1 = (1/Area)*(Isi[pos_x,pos_y+5] + Isi[pos_x+5,pos_y+11] - Isi[pos_x,pos_y+11] - Isi[pos_x+5,pos_y+5])
    E_black_2 = (1/Area)*(Isi[pos_x+5,pos_y] + Isi[pos_x+11,pos_y+5] - Isi[pos_x+5,pos_y+5] - Isi[pos_x+11,pos_y])
    
    sig_white_1 = E_white_1 - mu_white_1**2 
    sig_white_2 = E_white_2 - mu_white_2**2 
    sig_black_1 = E_black_1 - mu_black_1**2 
    sig_black_2 = E_black_2 - mu_black_2**2
    
    return sig_black_1 + sig_black_2 - sig_white_1 - sig_white_2

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
            Features[l] = rect_ver_24(Area_rhv_24, Ii, Isi, pos_x, pos_y)

            # Calcul des features pour les fenêtres rectangulaires horizontales 
            Features[l + 1] = rect_hor_24(Area_rhv_24, Ii, Isi, pos_x, pos_y)
            
            # Calcul des features pour les fenêtres tri-rectangulaires  
            Features[l + 2] = rect_triple_24(Area_triple_24, Ii, Isi, pos_x, pos_y)       
            
            # Calcul des features pour les fenêtres quadri-rectangulaires  
            Features[l + 3] = rect_quadruple_24(Area_quadruple_24, Ii, Isi, pos_x, pos_y)
            
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
            Features[l] = rect_ver_12(Area_rhv_12, Ii, Isi, pos_x, pos_y)

            # Calcul des features pour les fenêtres rectangulaires horizontales 
            Features[l + 1] = rect_hor_12(Area_rhv_12, Ii, Isi, pos_x, pos_y)
            
            # Calcul des features pour les fenêtres tri-rectangulaires  
            Features[l + 2] = rect_triple_12(Area_triple_12, Ii, Isi, pos_x, pos_y)       
            
            # Calcul des features pour les fenêtres quadri-rectangulaires  
            Features[l + 3] = rect_quadruple_12(Area_quadruple_12, Ii, Isi, pos_x, pos_y)
            
            pos_y += 4
            l += 4
        pos_x += 4
    
    # Normalisation des features 
    norm = np.sqrt(np.sum(Features ** 2))
    if norm != 0.0:
        Features = Features / norm
    
    return Features 

#%% Descripteurs LBP 

# --- Histogramme pour LPB ---
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
            val = int(Img_grid[i,j])
            hist[val] += 1 

    return hist 

@numba.njit
# --- Padding pour les bords des LBP ---
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
     
    padded_image = np.zeros((n + 2*pad, m + 2*pad), dtype = float)
    
    # Remplissage de l'image au centre de la matrice de padding
    
    padded_image[1:-1,1:-1] = img
    
    return padded_image 

#%% --- Fonction principale --- 

@numba.njit
def LBP(img):
    """
    Parameters
    ----------
    img : Array
        Image en teinte de gris dont on cherche le descripteur LBP

    Returns
    -------
    hist_img : Array
        Concaténation des histogrammes des features LBP de chaque grille de l'image
    """
    
    n,m = np.shape(img)

    # Convertion en uint8 pour une optimisation de la mémoire
    pict = padded_img(img = img, pad = 1)

    # Pour les conditions au bord, on ajoute juste un padding de zéro autour de l'image pour éviter d'introduire de l'information et donc de biaiser le descripteur
    LBP_img = np.zeros((n,m), dtype = "float64")

    # On suppose que les fenêtres utilisées sont carrées
    win_size = 3 

    # Python ne me laisse pas nommer la variable 2k pour 2^k, va savoir
    k2 = np.zeros(8, dtype = "float64")
    for i in range(8):
        k2[i] = 2**i

    for i in range(n):
        for j in range(m):
            # Définition de la fenêtre autour du pixel, du pixel centrale de la fenêtre et du vecteur représentant les pixels entourant le pixel : sens horaire en partant en haut à gauche du pixel
            window = pict[(i + 1 - win_size//2) : (i + win_size//2 + 2) , (j + 1 - win_size//2) : (j + win_size//2 + 2)]
            threshold = window[win_size//2,win_size//2]
            window_vect = np.zeros(win_size*win_size - 1)
            
            # Remplissage dudit vecteur 
            window_vect[:3] = window[0,:]
            window_vect[3] = window[1,2]
            window_vect[4:7] = window[2,::-1]
            window_vect[7] = window[1,0]
            
            # Convertion en binaire avec le threshold
            window_vect[window_vect < threshold] = 0
            window_vect[window_vect >= threshold] = 1
            
            # Calcul de la caractéristique LPB du pixel (i,j)
            LBP_img[i,j] = k2@window_vect
    
    # --- Histogrammes --- 

    # Partition de l'image en grilles de taille 16x16 

    grid_length = 16 
    nb_grid = n // grid_length
    nbins = 256

    # Chaque histogramme est de taille 256 donc au total on a nb_grid² * 256 ; Pour numba
    hist_img = np.empty(nb_grid * nb_grid * nbins, dtype=np.float64)

    idx = 0
    
    # Concaténation des histogrammes 
    for x in range(nb_grid):
        for y in range(nb_grid):
            Img_grid = LBP_img[x*grid_length : (x+1)*grid_length, y*grid_length : (y+1)*grid_length]
            
            feature = hist_LPB(Img_grid, grid_length, nbins)
            
            norm = np.sqrt(np.sum(feature * feature))
            if norm > 0.0:
                feature = feature / norm

            for k in range(nbins):
                hist_img[idx] = feature[k]
                idx += 1
            
    return hist_img 

#%% SVM hard margin avec UZAWA

@numba.jit
def Hard_margin(rho, it, tol, u, v, C):
    p = np.shape(u)[1]
    # Création du vecteur des multiplicateurs de Lagrange
    mu = np.ones((np.shape(C)[0],1))
    
    # Initialisation de l'erreur
    err = 10 
    
    # Initialisation du nombre d'itérations de l'algorithme
    k = 0 
    
    # Initilisation du vecteur 
    x = np.ones((np.shape(C)[1],1))
    
    # Boucle sur le nombre d'itération et sur l'erreur
    while k < it and err > tol: 
        x_cop = np.copy(x)
        x = (-1)*(C.T@mu) 
        mu = proj(mu + rho*(C@x + np.ones(((np.shape(C)[0]),1))))
        err = np.linalg.norm(x - x_cop)
        print(f"Itération de l'algorithme : {k} \nL'erreur comise est de : {err}")
        k += 1 
        if err > 1e6:
            print("Impossible de déterminer une solution.")
            break
        
    # Détermination de w 

    w = u@mu[:p] - v@mu[p:]

    # Détermination de b 

    mu1 = np.nonzero(mu[:p,0])[0]
    mu2 = np.nonzero(mu[p:,0])[0]
    b = 0 
    if mu1.size > 0 :
        for i in np.nditer(mu1):
            b = (1/(2*len(mu1)))*u[:,i].T@w + b
    if mu2.size > 0:
        for i in np.nditer(mu2):
            b = (1/(2*len(mu2)))*v[:,i].T@w + b  
    return mu, w, b

#%% Fonction de projection sur R+

@numba.jit
def proj(x):
    y = x
    for i in range(0, len(x)):
        y[i] = np.maximum(0,x[i])
    return y

#%% SVM soft margin avec UZAWA

@numba.jit
def Soft_margin(alpha, rho, it, tol, u, v, C):
    p = np.shape(u)[1]
    # Création du vecteur des multiplicateurs de Lagrange
    mu = np.ones((np.shape(C)[0],1))
    
    # Initialisation de l'erreur
    err = 10 
    
    # Initialisation du nombre d'itérations de l'algorithme
    k = 0 
    
    # Initilisation du vecteur 
    x = np.ones((np.shape(C)[1],1))
    
    # Boucle sur le nombre d'itération et sur l'erreur
    while k < it and err > tol: 
        x_cop = np.copy(x)
        x = (-1)*(C.T@mu) 
        mu = proj_alpha(mu + rho*(C@x + np.ones(((np.shape(C)[0]),1))),alpha)
        err = np.linalg.norm(x - x_cop)
        print(f"Itération de l'algorithme : {k} \nL'erreur comise est de : {err}")
        k += 1 
        if err > 1e6:
            print("Impossible de déterminer une solution.")
            break
        
    # Détermination de w 

    w = u@mu[:p] - v@mu[p:]

    # Détermination de b 

    mu1 = np.nonzero(mu[:p,0])[0]
    mu2 = np.nonzero(mu[p:,0])[0]
    b = 0 
    if mu1.size > 0 :
        for i in np.nditer(mu1):
            b = (1/(2*len(mu1)))*u[:,i].T@w + b
    if mu2.size > 0:
        for i in np.nditer(mu2):
            b = (1/(2*len(mu2)))*v[:,i].T@w + b  
    return mu, w, b

#%% Fonction de projection sur R+

@numba.jit
def proj_alpha(x,alpha):
    y = x
    for i in range(0, len(x)):
        y[i] = np.minimum(np.maximum(0,x[i]),alpha)
    return y

#%% Fonction qui permet de déterminer dans quelle partie du demi espace se trouve le point x

def f(x,w,b):
    if np.sign(w.T@x - b) > 0 :
        return 1 
    else: 
        return -1

#%% Noyau gaussien

@numba.jit
def gauss(x,y,sig):
    return np.exp(-(np.linalg.norm(x-y)**2)/(2*sig**2))

#%% Noyau polynômial

@numba.jit
def kern_poly(x, y, arg):
    """
    Calcule le noyau polynômial entre x et y :
        K(x, y) = (⟨x, y⟩ + c)^d

    - x, y : vecteurs ou matrices (lignes = vecteurs)
    - arg : arguments du noyau
    """
    c,d = arg 
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = (np.dot(x,y) + c)**d 
    elif (np.ndim(x) == 1 and np.ndim(y) > 1):
        print("2")
        result = (np.dot(y, x) + c)**d
    elif (np.ndim(x) > 1 and np.ndim(y) == 1):
        result = (np.dot(x,y) + c)**d
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = (np.dot(y, x) + c)**d
    return result

#%% Fonction pour déterminer les erreurs dans l'algorithme SMO 

@numba.jit 
def f_K_SMO(mu,Dy,b,k):
    return (mu.T@Dy@k)[0,0] + b

#%% Calcul de la matrice de Gramm pour le noyau K considéré

@numba.jit
def Gramm(K,X,arg):
    n = np.shape(X)[0]
    G = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            val = K(X[i,:],X[j,:],arg)
            G[i,j] = val
            G[j,i] = val
    return G

#%% SVM soft margin avec UZAWA à noyaux

@numba.jit
def Soft_margin_noyaux(alpha, rho, it, tol, u, v, C):
    p = np.shape(u)[0]
    q = np.shape(v)[0]
    # Création du vecteur des multiplicateurs de Lagrange
    mu = np.ones((np.shape(C)[0],1))
    
    # Initialisation de l'erreur
    err = 10 
    
    # Initialisation du nombre d'itérations de l'algorithme
    k = 0 
    
    # Initilisation du vecteur 
    x = np.ones((np.shape(C)[1],1))
    
    # Boucle sur le nombre d'itération et sur l'erreur
    while k < it and err > tol: 
        x_cop = np.copy(x)
        x = (-1)*(C.T@mu) 
        mu = proj_alpha(mu + rho*(C@x + np.ones((p+q,1))),alpha)
        err = np.linalg.norm(x - x_cop)
        print(f"Itération de l'algorithme : {k}")
        print("L'erreur comise est de :", err)
        k += 1 
        if err > 1e8:
            print("Impossible de déterminer une solution.")
            break
        
    # Détermination de a 

    a = C.T[:(p+q),:]@mu 

    b = C.T[-1,:]@mu 
    """# Détermination de b 

    mu1 = np.nonzero(mu[:p,0])[0]
    mu2 = np.nonzero(mu[p:,0])[0]
    b = 0 
    if mu1.size > 0 :
        for i in np.nditer(mu1):
            b = (1/(2*len(mu1)))*u[:,i].T@a + b
    if mu2.size > 0:
        for i in np.nditer(mu2):
            b = (1/(2*len(mu2)))*v[:,i].T@a + b  """
    return mu, a, b

#%% Fonction de décision associée pour l'algorithme d'Uzawa à noyaux 

def fD_uza(a,b,X,K,x,y):
    n = np.shape(X)[0]
    G = np.zeros((n,1))
    
    for i in range(n):
        G[i,0] = K(X[i,:],x)
    return (((a*y).T@G)[0] + b)[0]

#%% Fonction qui permet le calcul des metriques precision, recall et F1-score 

def metric_uza(y,a,b,X,K,x):
    n = len(y)
    confusion_matrix = np.zeros((2,2))
    for i in range(n):
        estimation = np.sign(fD_uza(a,b,X,K,x[i,:],y))
        if estimation > 0 and y[i] > 0:
            confusion_matrix[1,1] += 1 # Détection d'un true positive 
        elif estimation > 0 and y[i] < 0:
            confusion_matrix[0,1] += 1 # Détection d'un false positive
        elif estimation < 0 and y[i] > 0:
            confusion_matrix[1,0] += 1 # Détection d'un false negative 
        elif estimation < 0 and y[i] < 0:
            confusion_matrix[0,0] += 1 # Détection d'un true negative 
            
    if (confusion_matrix[1,1] != 0) or (confusion_matrix[0,1] != 0):
        precision = confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[0,1])
    else:
        precision = 0 
        
    if (confusion_matrix[1,1] != 0) or (confusion_matrix[1,0] != 0):
        recall = confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[1,0])
    else:
        recall = 0 
        
    if recall == 0 and precision == 0:
        F1 = 0
    else:
        F1 = 2*precision*recall/(precision+recall) 
    accuracy = (confusion_matrix[0,0]+confusion_matrix[1,1])/n 
    return accuracy, precision, recall, F1,confusion_matrix

#%% Fonction qui permet d'afficher la caméra pour tester en temps réel l'algorithme
"""
def cam(a,b,kernel,X,y):
    # Ouvrir la caméra (0 si caméra principale)
    cap = cv2.VideoCapture(0)
    
    # Vérification de l'ouverture de la caméra
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        exit()
    
    # Coordonnées (x, y) pour positionner le texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Boucle de capture vidéo
    while True:
        # Lecture de l'image
        ret, frame = cap.read()
        # Vérifier si la lecture de l'image est réussie
        if not ret:
            print("Erreur: Impossible de lire l'image depuis la caméra.")
            break
        
        # Capture de l'image pour la fonction test 
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_test = cv2.resize(img,(32,32)).reshape(-1, order = 'C').astype("float64")
        #img_test = cv2.normalize(img_test,None, 0, 1, cv2.NORM_MINMAX)
        img_test = img_test*fac + 0.01
        estimation = fD_uza(a,b,X,kernel,img_test,y)
        print(estimation)
        test = np.sign(estimation)
        
        if test == 1 :
            # position où on écrit le texte
            position = (int(0.36*frame.shape[1]),int(0.10*frame.shape[0])) 
            text = "Bienvenue !"
            font_thickness = 2
            font_scale = 1
            font_color = (0, 255, 0)  # Vert
            cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)
        elif test == -1:
            # position où on écrit le texte
            position = (int(0.25*frame.shape[1]),int(0.10*frame.shape[0])) 
            font_thickness = 2
            font_scale = 1
            text = "Visage non reconnu"
            font_color = (0, 0, 255)  # Rouge
            cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)
        
        # Afficher l'image en temps réel
        cv2.imshow('Camera en temps reel', frame)
        
        # Attendre 1 milliseconde et vérifier si l'utilisateur appuie sur la touche echap pour quitter
        if cv2.waitKey(1) == 27 :
            break
"""
#%% SVM soft margin algorithme SMO simplifié 

def SMO_simplified(alpha, tol, N, X, Y, K):
    """
    Parameters
    ----------
    alpha : Real
        Marge que l'on autorise à notre soft margin
    tol : Real
        Tolérance admise pour notre algorithme 
    N : Int
        Nombre maximum de fois où l'on peut itérer sur les mu sans modification
    X : Array
        Matrice contenant les images d'entraînement
    Y : Array
        Labels associés à chacune de ces images (sortie recherchée pour notre SVM) 
    K : Function 
        Indique quel noyau on utilise dans l'algorithme de résolution

    Returns
    -------
    mu : Array
        Multiplicateurs de Lagrange pour déterminer la fonction de décision 
        
    b : Real 
        Threshold de la SVM
    """
    # Variables générales de l'algorithme 
    
    # Nombre d'exemples d'entraînement 
    n = np.shape(X)[0]
    
    # Multiplicateurs de Lagrange 
    mu = np.zeros((n,1)) 
    
    # Threshold de la SVM
    b = 0
    
    # Nombre de boucle sur les données d'entraînement sans modifier de mu (se réinitialise à chaque fois que l'on modifie un mu)
    k = 0 
    
    # Matrice diagonale des valeurs vraies utilisée pour calculer les erreurs 
    Dy = np.diag(Y.reshape(-1,))
    
    # Matrice de Gramm utilisée pour calculer les erreurs   
    G = Gramm(K,X)
    
    # Boucle globale sur le nombre maximum de fois que l'on s'autorise à boucler sur les données d'entraînement sans modifier de mu
    while k < N:
        changed_mu = 0
        # On parcourt les n exemples 
        for i in range(n):
            
            # On récupère le multiplicateur de lagrange associé à la première donnée d'entraînement (i) ainsi que sa classe réelle
            
            mu1 = mu[i,0]
            y1 = Y[i,0]
            
            # Calcul de l'erreur sur cet exemple
            
            E1 = f_K_SMO(mu,Dy,b,G[i,:].reshape(-1,1)) - y1
            
            # Pas encore compris celle-là 
            if (y1*E1 < -tol and mu1 < alpha) or (y1*E1 > tol and mu1 > 0):
            
                # Génération d'un nombre entier aléatoire entre 1 et n différent de i 
                
                j = np.random.randint(1,n)
                while j == i:
                    j = np.random.randint(1, n)
                
                # On récupère le multiplicateur de Lagrange associé à la deuxième donnée d'entraînement (j) ainsi que sa classe réelle
                
                mu2 = mu[j,0]
                y2 = Y[j,0]
                
                # Calcul de l'erreur sur cet exemple 
                
                E2 = f_K_SMO(mu,Dy,b,G[j,:].reshape(-1,1)) - y2
                
                # Discrimination en fonction de si les exemples sont de la même classe ou non
                if y1 == y2:
                    L = np.maximum(0,mu1 + mu2 - alpha)
                    H = np.minimum(alpha,mu1+mu2)
                elif y1 != y2:
                    L = np.maximum(0, mu2 - mu1)
                    H = np.minimum(alpha, alpha + mu2 - mu1)
                
                # Vérification que l'on fera du progrès pour cet exemple i ou non, si non, on le change
                
                if L == H:
                    continue 
                eta = G[i,i] + G[j,j] - 2*G[i,j]
                if eta <= 0:
                    continue 
                
                # Calcul du nouveau multiplicateur de Lagrange pour la deuxième donnée d'entraînement
                
                mu2_new = mu2 + (y2*(E1-E2))/eta 
                if mu2_new < L:
                    mu2_new_clipped = L 
                elif mu2_new > H:
                    mu2_new_clipped = H
                else:
                    mu2_new_clipped = mu2_new 
                if np.abs(mu2_new_clipped - mu2) < 1e-5:
                    continue # Change de i, on ne fait pas de progrès 
                
                # Calcul du nouveau multiplicateur de Lagrange pour la première donnée d'entraînement
                
                mu1_new = mu1 + y1*y2*(mu2-mu2_new_clipped)
                
                # Mise à jour de b 
                
                b1 = b - E1 - y1*(y1*y2*(mu2-mu2_new_clipped))*G[i,i] - y2*(mu2_new_clipped - mu2)*G[i,j]
                b2 = b - E2 - y1*(y1*y2*(mu2-mu2_new_clipped))*G[i,j] - y2*(mu2_new_clipped - mu2)*G[j,j]
                if (mu1 < alpha and mu1 > 0) or (mu2 < alpha and mu2 > 0):
                    b = b1
                else:
                    b = (b1+b2)/2
                
                # Mise à jour des multiplicateurs de Lagrange 
                
                mu[i,0] = mu1_new 
                mu[j,0] = mu2_new_clipped
                
                # Actualisation pour réinitialiser k au cas où l'on a changé des multiplicateurs de Lagrange 
              
                changed_mu += 1
                
        if changed_mu == 0:
            k += 1 
        else:
            k = 0
            
    return mu,b

#%% Fonction de décision pour SVM soft margin à noyaux pour les images x présentes dans la base de donnée 

def f_D_data(mu,y,b,G):
    """
    Parameters
    ----------
    x : Array
        Image pour laquelle on souhaite tester notre classification 
    mu : Array
        Multiplicateurs de Lagrange des vecteurs supports obtenus par l'algorithme de SMO
    y : Array
        Vecteurs des vraies labels des données d'entraînements 
    b : Real
        Threshold pour la SVM (ordonnée à l'origine)
    G : Array
        Vecteur de la matrice de Gramm correspondant à la donnée x

    Returns
    -------
    Real
        Résultation de la classification obtenue 
    """
    return ((mu*y).T@G) - b

#%% Fonction de décission pour SVM soft margin à noyaux pour des images x quelconques

def f_D(x,X,mu,y,K,b):
    """
    Parameters
    ----------
    x : Array
        Image dont on souhaite déterminer la classification 
    X : Matrix
        Ensemble des données d'entraînement associé aux vecteurs supports
    mu : Array
        Multiplicateurs de Lagrange des vecteurs supports obtenus par l'algorithme de SMO
    y : Array
        Vecteurs des vraies labels des données d'entraînements 
    K : Function
        Noyau à utiliser
    b : Real
        Threshold pour la SVM (ordonnée à l'origine)

    Returns
    -------
    Real
        Résultat de la classification obtenue
    """
    # Nombre de vecteurs supports 
    n = np.shape(mu)[0] 
    
    # Vecteur qui contiendra les résultats du noyau entre x et les données 
    G = np.zeros((n,1)) 
    
    for i in range(n):
        G[i,0] = K(x,X[i,:],7)
        
    return ((mu*y).T@G)[0,0] - b

#%% Fonction qui permet le calcul des metriques precision, recall et F1-score 

def metric(model, x_test, y_test):
    n = len(x_test)
    confusion_matrix = np.zeros((2,2))
    for i in range(n):
        estimation = np.sign(SVMC.decision_function(model, x_test[i,:]))
        if estimation > 0 and y_test[i] > 0:
            confusion_matrix[1,1] += 1 # Détection d'un true positive 
        elif estimation > 0 and y_test[i] < 0:
            confusion_matrix[0,1] += 1 # Détection d'un false positive
        elif estimation < 0 and y_test[i] > 0:
            confusion_matrix[1,0] += 1 # Détection d'un false negative 
        elif estimation < 0 and y_test[i] < 0:
            confusion_matrix[0,0] += 1 # Détection d'un true negative 
    try:
        precision = confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[0,1])
        recall = confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[1,0])
        F1 = 2*precision*recall/(precision+recall) 
        accuracy = (confusion_matrix[0,0]+confusion_matrix[1,1])/n 
        
        return accuracy, precision, recall, F1,confusion_matrix
    
    except ValueError:
        print("Division by zero occured in the confusion matrix, change the model")
        return None 

#%% Affichage de la métrique Accuracy

def Affiche_acc(y,mu,b,K):
    n = len(y)
    accuracy = np.zeros((n,1))
    for i in range(n):
        estimation = np.sign(f_D_data(mu,y,b,K[:,i].reshape(-1,1)))[0]
        if estimation == y[i,0]:
            accuracy[i,0] = 1 
    
    hauteur = 50 
    largeur = n//hauteur  
    acc_reshape = accuracy[:50,0].reshape(-1,1)
    print(np.shape(acc_reshape))
    for i in range(1,largeur):
        acc_reshape = np.concatenate((acc_reshape, accuracy[50*i:(50*(i+1)),0].reshape(-1,1)), axis = 1)
    print(np.shape(acc_reshape))
    plt.imshow(acc_reshape)

#%% Fonction that crops and resizes the images for the preprocessing of the data

def crop_and_resize(img, w, h):
        im_h, im_w = img.shape[:2]
        res_aspect_ratio = w/h
        input_aspect_ratio = im_w/im_h

        if input_aspect_ratio > res_aspect_ratio:
            im_w_r = int(input_aspect_ratio*h)
            im_h_r = h
            img = cv2.resize(img, (im_w_r , im_h_r))
            x1 = int((im_w_r - w)/2)
            x2 = x1 + w
            img = img[:, x1:x2]
        if input_aspect_ratio < res_aspect_ratio:
            im_w_r = w
            im_h_r = int(w/input_aspect_ratio)
            img = cv2.resize(img, (im_w_r , im_h_r))
            y1 = int((im_h_r - h)/2)
            y2 = y1 + h
            img = img[y1:y2, :]
        if input_aspect_ratio == res_aspect_ratio:
            img = cv2.resize(img, (w, h))

        return img

#%% 
@numba.jit
def ACPK(X,K):
    """
    Parameters
    ----------
    X : Array
        Données sur lesquelles on souhaite appliquer l'ACP
    K : Array
        Matrice de Gramm associée aux données et au noyau considéré pour appliquer l'ACP

    Returns
    -------
    P : Array
        Matrice de projection des données
    Sig : Array
        Matrice contenant des valeurs propres
        
    """
    # Récupération des dimensions des données
    m,d = np.shape(X)
    
    # Calcul de la matrice K_chap
    U = (1/m)*np.ones((m,m))
    I = np.eye(m)
    K_chap = (I-U)@K@(I-U)
    
    U,Sig,Vt = np.linalg.svd(K_chap, hermitian = True)
    
    # Règle de Kaiser pour le choix de k 
    
    # On récupère le nombre d de lignes de la matrice
    d = np.shape(U)[0]
    
    # On calcule la somme des valeurs propres de la matrice de covariance C
    s = np.sum(Sig)
    
    # Initialisation de la règle de Kaiser
    lambda_k = 0
    k = 1
    
    # Incrémentation de k tant que la règle de Kaiser n'est pas respectée
    while lambda_k < (1/d)*s:
        lambda_k = Sig[k-1]
        k += 1
    
    Vk = U[:,:k]
    
    P = Vk@np.diag(np.sqrt(Sig[:k]))    
    
    return P,Sig,Vk

#%% Fonction qui permet la classification d'image par 

@numba.jit 
def Classification_Haar(Img, model,arg):
    m,n = np.shape(Img)
    
    window = 12 
    while window < m or window < n:
        kx = int((3*m)/window - 2)
        ky = int((3*n)/window - 2)
        pos_x = 0
        for i in range(kx):
            pos_y = 0
            for j in range(ky):
                Img_fetch = cv2.resize(Img[(pos_x):(pos_x+window),(pos_y):(pos_y+window)],(64,64)) 
                Features = Haar_features(Img_fetch)
                test = np.abs(SVMC.decision_function2(model.mu, model.y, model.X, Features, model.b, model.kernel, arg))
                if test > 0:
                    return test, True 
                pos_y += int(window/3) 
            pos_x += int(window/3)
        window = round(window*1.25)
        return False 

#%% Fonction qui permet d'afficher la caméra pour tester en temps réel l'algorithme

def cam(model, config):
    # Ouvrir la caméra (0 si caméra principale)
    cap = cv2.VideoCapture(0)
    
    # Vérification de l'ouverture de la caméra
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        exit()
    
    # Coordonnées (x, y) pour positionner le texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Initialiser le timer
    last_check = time.time()
    intervalle = 0.5  # 0.5 secondes
    value = 0
    test = 0
    text = ""
    
    # Boucle de capture vidéo
    while True:
        # Lecture de l'image
        ret, frame = cap.read()
        # Vérifier si la lecture de l'image est réussie
        if not ret:
            print("Erreur: Impossible de lire l'image depuis la caméra.")
            break
        
        hauteur, largeur, _ = frame.shape

        # Dimensions du rectangle central
        hauteur_rect = 450 
        largeur_rect = 300
        x_centre = largeur // 2
        y_centre = hauteur // 2
        x1 = x_centre - largeur_rect // 2
        y1 = y_centre - hauteur_rect // 2
        x2 = x_centre + largeur_rect // 2
        y2 = y_centre + hauteur_rect // 2

        # Dessiner un rectangle bleu (épaisseur 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Dessiner les axes X et Y en bleu à l'intérieur du rectangle
        # Axe Y (vertical)
        cv2.line(frame, (x_centre, y1), (x_centre, y2), (255, 0, 0), 1)
        # Axe X (horizontal)
        cv2.line(frame, (x1, y_centre), (x2, y_centre), (255, 0, 0), 1)
        
        # Ajouter le texte "placez votre tête ici"
        texte = "Placez votre tete ici"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        epaisseur = 1
        couleur = (255, 0, 0)  # Bleu

        # Obtenir les dimensions du texte
        (text_width, text_height), _ = cv2.getTextSize(texte, font, font_scale, epaisseur)
        text_x = x_centre - text_width // 2
        text_y = y2 - 25  # 50 pixels au-dessus du bas du rectangle

        # Afficher le texte
        cv2.putText(frame, texte, (text_x, text_y), font, font_scale, couleur, epaisseur, cv2.LINE_AA)

        # Extraire la portion centrale de l'image
        frame_centre = frame[y1:y2, x1:x2]
        
        # Vérifier si 0.5s sont passées depuis le dernier test
        if time.time() - last_check >= intervalle:
            last_check = time.time()
    
            # Traitement SVM
            img = cv2.cvtColor(frame_centre, cv2.COLOR_BGR2GRAY)
            img_test = preprocess_image_cam(img).astype("float64")
            x = config(img_test, None)
            value = SVMC.decision_function(model, x)
            test = np.sign(value)
            print(value)

        # Afficher les résultats du dernier test connu
        if test == 1:
            position = (int(0.36 * frame.shape[1]), int(0.10 * frame.shape[0]))
            text = "Bienvenue !"
            font_color = (0, 255, 0)
        elif test == -1:
            position = (int(0.25 * frame.shape[1]), int(0.10 * frame.shape[0]))
            text = "Visage non reconnu"
            font_color = (0, 0, 255)
        else:
            text = ""
            font_color = (255, 255, 255)

        if text:
            cv2.putText(frame, text, position, font, 1, font_color, 2)
        
        # Afficher l'image en temps réel
        cv2.imshow('Camera en temps reel', frame)
        
        # Attendre 1 milliseconde et vérifier si l'utilisateur appuie sur la touche echap pour quitter
        if cv2.waitKey(1) == 27 :
            break
        
    cap.release()
    cv2.destroyAllWindows() 

#%% SVM soft margin à partir de l'algorithme d'uzawa à noyaux 

def Soft_margin_K(alpha, roh, it, tol, u, v, C):
    """
    Cette fonction est la même que la fonction soft margin à la différence que les vecteurs u, v et la matrice C à 
    donner en entrée de cette dernières continennent les données transformées par le noyau
    Parameters
    ----------
    alpha : TYPE
        DESCRIPTION.
    roh : TYPE
        DESCRIPTION.
    it : TYPE
        DESCRIPTION.
    tol : TYPE
        DESCRIPTION.
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    C : TYPE
        DESCRIPTION.

    Returns
    -------
    mu : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    """
    p = np.shape(u)[1]
    # Création du vecteur des multiplicateurs de Lagrange
    mu = np.ones((np.shape(C)[0],1))
    
    # Initialisation de l'erreur
    err = 10 
    
    # Initialisation du nombre d'itérations de l'algorithme
    k = 0 
    
    # Initilisation du vecteur 
    x = np.ones((np.shape(C)[1],1))
    
    # Boucle sur le nombre d'itération et sur l'erreur
    while k < it and err > tol: 
        x_cop = np.copy(x)
        x = (-1)*(C.T@mu) 
        mu = proj_alpha(mu + rho*(C@x + np.ones(((np.shape(C)[0]),1))),alpha)
        err = np.linalg.norm(x - x_cop)
        print(f"Itération de l'algorithme : {k} \nL'erreur comise est de : {err}")
        k += 1 
        if err > 1e6:
            print("Impossible de déterminer une solution.")
            break
        
    # Détermination de w 

    w = u@mu[:p] - v@mu[p:]

    # Détermination de b 

    mu1 = np.nonzero(mu[:p,0])[0]
    mu2 = np.nonzero(mu[p:,0])[0]
    b = 0 
    if mu1.size > 0 :
        for i in np.nditer(mu1):
            b = (1/(2*len(mu1)))*u[:,i].T@w + b
    if mu2.size > 0:
        for i in np.nditer(mu2):
            b = (1/(2*len(mu2)))*v[:,i].T@w + b  
    return mu, w, b

#%% 

#@numba.jit
def center(X):
    """
    Parameters
    ----------
    X : Array
        La matrice X est la matrice des réalisations de taille n,m

    Returns
    -------
    Xr : Array
        La matrice Xr est la matrice centrée de X

    """
    n,m = np.shape(X) # Récupération des dimensions de X
    ones = np.ones((n,m)) # Création d'un vecteur constitué de 1 de la taille de X
    Xm = np.mean(X, axis = 1).reshape(-1,1) # Détemination de la moyenne de la matrice X sous forme de vecteur
    X_tilde = ones*Xm # Détermination de la matrice des moyennes des Xi
    
    Xr = X - X_tilde # Calcul de la matrice centrée Xr
    return Xr
            
#%% Test de whitening sur les données

#@numba.jit
def whitening(Xr):
    """
    Parameters
    ----------
    Xr : Array
        Matrice aléatoire centrée

    Returns
    -------
    Z : Array
        Matrice de réalisation blanchie
        
    """
    m,n = np.shape(Xr) # Récupérations des dimensions de Xr
    Cov = (1/(n-1))*Xr@Xr.T # Calcul de la matrice de covariance de X 
    
    U,pho,VT = np.linalg.svd(Cov, hermitian = True) # Décomposition en valeurs singulières de la matrice de covariance inverse
    
    D_12 = np.linalg.pinv(np.diag(pho))**(1/2)
    
    Z = D_12@U.T@Xr # Détermination de la matrice de réalisation blanchie 
    return Z


#%% Fonction de moindre carrés à noyaux 

def Moindre_carre_K(rho, it, tol, K, B, alpha):
    lb,cb = np.shape(B) # Récupération de la taille des données
    lK,cK = np.shape(K)
    
    L = np.ones((lK,cb)) # Initialisation du vecteur des coefficients de Lagrange
    
    ones = np.ones((lK,cb)) # Création d'une matrice constitué que de 1
    
    A = np.ones((cK,cb))
    
    err = 10 # Initialisation de l'erreur
    
    k = 0 # Initialisation du nombre d'itérations
    
    while err > tol and k < it:
        T = np.copy(A)
        
        A = np.linalg.inv(K)@K.T@(L*B)
        
        L = L + rho*(-(K@A)*B+ones) # Mise à jour de Lambda
        
        L = np.minimum(np.maximum(L,0), alpha) # Projection de Lambda sur [0, marge]
        
        err = np.linalg.norm(A - T) # Actualisation de l'erreur
        
        k += 1
        
        print(f"Itération de l'algorithme : {k} pour une erreur de : {err}")
            
    return A,L
    
   
#%% Fonction qui permet d'afficher la caméra pour tester en temps réel l'algorithme
"""
def cam(w,b):
    # Ouvrir la caméra (0 si caméra principale)
    cap = cv2.VideoCapture(0)
    
    # Vérification de l'ouverture de la caméra
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        exit()
    
    # Coordonnées (x, y) pour positionner le texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Boucle de capture vidéo
    while True:
        # Lecture de l'image
        ret, frame = cap.read()
        # Vérifier si la lecture de l'image est réussie
        if not ret:
            print("Erreur: Impossible de lire l'image depuis la caméra.")
            break
        
        # Initialisation des variables pour gérer le texte
        text_blue = "Veuillez lancer le programme de reconnaissance faciale"
        font_thickness = 2
        font_scale = 1
        font_color_blue = (255, 0, 0)  # Bleu
        
        # Centrage et découpage du texte bleu pour retour à la ligne
        (text_width, text_height), _ = cv2.getTextSize(text_blue, font, font_scale, font_thickness)
        max_text_width = frame.shape[1] - 40  # Marge de 20 pixels de chaque côté
        wrapped_text = []
        words = text_blue.split(' ')
        current_line = ""
        
        for word in words:
            if cv2.getTextSize(current_line + word + " ", font, font_scale, font_thickness)[0][0] <= max_text_width:
                current_line += word + " "
            else:
                wrapped_text.append(current_line.strip())
                current_line = word + " "
        wrapped_text.append(current_line.strip())

        # Dessiner le texte bleu centré si aucun texte rouge ou vert n'est affiché
        if cv2.waitKey(1) != 112 and cv2.waitKey(1) != 27:
            y0 = int(0.10 * frame.shape[0])
            for i, line in enumerate(wrapped_text):
                text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                x = int((frame.shape[1] - text_size[0]) / 2)  # Centrer horizontalement
                y = y0 + i * text_size[1] + i * 10  # Ajuster l'espacement entre les lignes
                cv2.putText(frame, line, (x, y), font, font_scale, font_color_blue, font_thickness)
        
        img_moy = np.zeros((480,640))
        N = 1000 
        k = 0 
        test = int
        if cv2.waitKey(1) == 112:
            while k < N:
                
                # Capture de l'image pour en faire les moyennes
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float64")
                img_moy += img 
                k+= 1
                
                # Afficher l'image en temps réel
                cv2.imshow('Camera en temps reel', frame)
                
            img_moy = img_moy/N
            
            centered_frame = center(img_moy)
            whitened_frame = whitening(centered_frame)
            
            img_test = cv2.resize(whitened_frame,(32,32)).reshape(-1, order = 'C').astype("float64")
            img_test = img_test*fac + 0.01 
            test = f(img_test,w,b)
            
            
            img_test = cv2.resize(img_moy,(32,32)).reshape(-1, order = 'C').astype("float64")
            img_test = img_test*fac + 0.01 
            test = f(img_test,w,b)
        
        centered_frame = center(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        whitened_frame = whitening(centered_frame)
        
        img_test = cv2.resize(whitened_frame,(32,32)).reshape(-1, order = 'C').astype("float64")
        img_test = img_test*fac + 0.01 
        test = f(img_test,w,b)
        
        if test == 1 :
            # position où on écrit le 
            position = (int(0.36*frame.shape[1]),int(0.10*frame.shape[0])) 
            text = "Bienvenue !"
            font_thickness = 2
            font_scale = 1
            font_color = (0, 255, 0)  # Vert
            cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)
        elif test == -1:
            # position où on écrit le texte
            position = (int(0.25*frame.shape[1]),int(0.10*frame.shape[0])) 
            font_thickness = 2
            font_scale = 1
            text = "Visage non reconnu"
            font_color = (0, 0, 255)  # Rouge
            cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)
        
        # Afficher l'image en temps réel
        cv2.imshow('Camera en temps reel', frame)
        
        # Attendre 1 milliseconde et vérifier si l'utilisateur appuie sur la touche echap pour quitter
        if cv2.waitKey(1) == 27 :
            break
"""
#%% Fonction qui permet d'afficher la caméra pour tester en temps réel l'algorithme
"""
def cam(w,b):
    # Ouvrir la caméra (0 si caméra principale)
    cap = cv2.VideoCapture(0)
    
    # Vérification de l'ouverture de la caméra
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        exit()
    
    # Coordonnées (x, y) pour positionner le texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    test = int
    
    font_thickness = 2
    font_scale = 1
    
    # Initialisation des variables pour gérer le texte bleu
    text_blue = "Veuillez lancer le programme de reconnaissance faciale"
    blue = (255, 0, 0)
    
    # Initialisation des variables pour gérer le texte rouge
    text_red = "Visage non reconnu"
    red = (0, 0, 255)  # Rouge
    
    # Initialisation des variables pour gérer le texte vert
    text_green = "Bienvenue !"
    green = (0, 255, 0)  # Vert
    
    # Boucle de capture vidéo
    while True:
        # Lecture de l'image
        ret, frame = cap.read()
        # Vérifier si la lecture de l'image est réussie
        if not ret:
            print("Erreur: Impossible de lire l'image depuis la caméra.")
            break
        
        # Centrage et découpage du texte bleu pour retour à la ligne
        (text_width, text_height), _ = cv2.getTextSize(text_blue, font, font_scale, font_thickness)
        max_text_width = frame.shape[1] - 40  # Marge de 20 pixels de chaque côté
        wrapped_text = []
        words = text_blue.split(' ')
        current_line = ""
        
        for word in words:
            if cv2.getTextSize(current_line + word + " ", font, font_scale, font_thickness)[0][0] <= max_text_width:
                current_line += word + " "
            else:
                wrapped_text.append(current_line.strip())
                current_line = word + " "
        wrapped_text.append(current_line.strip())

        # Dessiner le texte bleu centré si aucun texte rouge ou vert n'est affiché
        if cv2.waitKey(1) != 112 and cv2.waitKey(1) != 27:
            y0 = int(0.10 * frame.shape[0])
            for i, line in enumerate(wrapped_text):
                text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                x = int((frame.shape[1] - text_size[0]) / 2)  # Centrer horizontalement
                y = y0 + i * text_size[1] + i * 10  # Ajuster l'espacement entre les lignes
                cv2.putText(frame, line, (x, y), font, font_scale, blue, font_thickness)
        
        img_moy = np.zeros((480,640,3))
        centered_frame = np.copy(img_moy)
        whitened_frame = np.copy(img_moy)
        N = 1000 
        k = 0 
        
        if cv2.waitKey(1) == 112:
            while k < N:
                
                # Capture de l'image pour en faire les moyennes                
                img_moy += frame 
                k+= 1
                
                # Afficher l'image en temps réel
                cv2.imshow('Camera en temps reel', frame)
                
            for i in range(3):
                centered_frame[:,:,i] = center(img_moy[:,:,i])
                whitened_frame[:,:,i] = whitening(centered_frame[:,:,i])
            
            img_test = cv2.resize(cv2.cvtColor(whitened_frame.astype("uint8"), cv2.COLOR_BGR2GRAY),(32,32)).reshape(-1, order = 'C').astype("float64")
            img_test = img_test*fac + 0.01 
            test = f(img_test,w,b)
            """
"""
            img_test = cv2.resize(img_moy,(32,32)).reshape(-1, order = 'C').astype("float64")
            img_test = img_test*fac + 0.01 
            test = f(img_test,w,b)"""
        
"""centered_frame = center(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        whitened_frame = whitening(centered_frame)
        
        img_test = cv2.resize(whitened_frame,(32,32)).reshape(-1, order = 'C').astype("float64")
        img_test = img_test*fac + 0.01 
        test = f(img_test,w,b)"""
"""
        if test == 1 :
            # position où on écrit le 
            position_green = (int(0.36*frame.shape[1]),int(0.10*frame.shape[0])) 
            cv2.putText(frame, text_green, position_green, font, font_scale, green, font_thickness)
        elif test == -1:
            position_red = (int(0.25*frame.shape[1]),int(0.10*frame.shape[0])) 
            cv2.putText(frame, text_red, position_red, font, font_scale, red, font_thickness)
        
        # Afficher l'image en temps réel
        cv2.imshow('Camera en temps reel', frame)
        
        # Attendre 1 milliseconde et vérifier si l'utilisateur appuie sur la touche echap pour quitter
        if cv2.waitKey(1) == 27 :
            break
"""