# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:25:00 2024

@author: eliot
"""

import cv2
import numpy as np
import os
import SVM_lib as SVM
import SVM_class as SVMC 
import matplotlib.pyplot as plt 
from PIL import Image
import pickle
import time as t 

"""
#%% Génération de la base de données pour les SVM avec train_data_1

# Création de la liste qui va contenir les matrices des images

Liste_faces = []

files = os.listdir("Train_data_1")
for i in range(1,len(files) + 1):
    Liste_faces.append(cv2.imread(f"Train_data_1/{i}.jpg",0).astype("float64"))
    Liste_faces[i-1] = cv2.resize(Liste_faces[i-1],(240,180))

taille_img_1 = 240
taille_img_2 = 180

train_data = np.zeros((taille_img_1*taille_img_2 + 1,len(files)))

for i in range(len(Liste_faces)):
    train_data[1:,i] = Liste_faces[i].reshape(-1, order = 'C')
    
for i in range(17):
    train_data[0,i] = 1
   
for i in range(17,41):
    train_data[0,i] = -1
    
for i in range(41,49):
    train_data[0,i] = 1
    
train_data[0,-1] = -1

np.save("SVM-Noyaux/Hard_margin/train_data.npy", train_data, allow_pickle = True)

#%% Création des vecteurs u et v associés à train_data_1

train_imgs = np.asfarray(train_data[1:,:])*fac + 0.01

values = train_data[0,:]
indice_u = np.where(values == 1)
indice_v = np.where(values != 1)

p = len(indice_u[0])
q = len(indice_v[0])

u = np.zeros((np.shape(train_data[1:,0])[0],p))
v = np.zeros((np.shape(train_data[1:,0])[0],q))

for i in range(p):
    u[:,i] = train_imgs[:,indice_u[0][i]]

for i in range(q):    
    v[:,i] = train_imgs[:,indice_v[0][i]]

#np.save("SVM-Noyaux/Hard_margin/u.npy", u, allow_pickle = True)
#np.save("SVM-Noyaux/Hard_margin/v.npy", v, allow_pickle = True)

#%% Test des fonctions pour train_data_1

rho = 0.000001 
it = 15000 
tol = 1e-6
u = np.load("SVM-Noyaux/Hard_margin/u.npy", allow_pickle = True)
v = np.load("SVM-Noyaux/Hard_margin/v.npy", allow_pickle = True)
p = np.shape(u)[1]
q = np.shape(v)[1]

X = np.concatenate((-u.T,v.T), axis = 0)
ones = np.block([[np.ones((p,1))],[-np.ones((q,1))]])
C = np.concatenate((X,ones), axis = 1)

mu, w, b = SVM.Hard_margin(rho, it, tol, u, v, C)

#%% Test de la fonction d'affichage pour train_data_1

w = np.load("SVM-Noyaux/Hard_margin/w_1.npy", allow_pickle = True)
b = np.load("SVM-Noyaux/Hard_margin/b_1.npy", allow_pickle = True)

SVM.cam(w,b)

#%% Test des fonctions de soft margin pour train_data_1

rho = 0.000001 
it = 15000 
tol = 1e-6
alpha = 0.05
u = np.load("SVM-Noyaux/Hard_margin/u.npy", allow_pickle = True)
v = np.load("SVM-Noyaux/Hard_margin/v.npy", allow_pickle = True)
p = np.shape(u)[1]
q = np.shape(v)[1]

X = np.concatenate((-u.T,v.T), axis = 0)
ones = np.block([[np.ones((p,1))],[-np.ones((q,1))]])
C = np.concatenate((X,ones), axis = 1)

mu, w, b = SVM.Soft_margin(alpha, rho, it, tol, u, v, C)

#%% Génération de la base de données pour train_data_2 


# Création de la liste qui va contenir les matrices des images

Liste_faces = []

files = os.listdir("Train_data_2")
for i in range(1,len(files) + 1):
    Liste_faces.append(cv2.imread(f"Train_data_2/{i}.jpg",0).astype("float64"))
    Liste_faces[i-1] = cv2.resize(Liste_faces[i-1],(240,180))

taille_img_1 = 240
taille_img_2 = 180

train_data = np.zeros((taille_img_1*taille_img_2 + 1,len(files)))

for i in range(len(Liste_faces)):
    train_data[1:,i] = Liste_faces[i].reshape(-1, order = 'C')

for i in range(1,42):
    train_data[0,i] = 1
    
for i in range(42, len(Liste_faces)):
    train_data[0,i] = -1

np.save("SVM-Noyaux/Hard_margin/train_data_2.npy", train_data, allow_pickle = True)

#%% Création des vecteurs u et v associés à train_data_2

train_imgs = np.asfarray(train_data[1:,:])*fac + 0.01

values = train_data[0,:]
indice_u = np.where(values == 1)
indice_v = np.where(values != 1)

p = len(indice_u[0])
q = len(indice_v[0])

u = np.zeros((np.shape(train_data[1:,0])[0],p))
v = np.zeros((np.shape(train_data[1:,0])[0],q))

for i in range(p):
    u[:,i] = train_imgs[:,indice_u[0][i]]

for i in range(q):    
    v[:,i] = train_imgs[:,indice_v[0][i]]

#%% Test de la fonction SVM pour train_data_2

rho = 0.000001 
it = 15000 
tol = 1e-6
u = np.load("SVM-Noyaux/Hard_margin/u_2.npy", allow_pickle = True)
v = np.load("SVM-Noyaux/Hard_margin/v_2.npy", allow_pickle = True)
p = np.shape(u)[1]
q = np.shape(v)[1]

X = np.concatenate((-u.T,v.T), axis = 0)
ones = np.block([[np.ones((p,1))],[-np.ones((q,1))]])
C = np.concatenate((X,ones), axis = 1)

mu, w, b = SVM.Hard_margin(rho, it, tol, u, v, C)

np.save("SVM-Noyaux/Hard_margin/w_2.npy", w, allow_pickle = True)
np.save("SVM-Noyaux/Hard_margin/b_2.npy", b, allow_pickle = True)

#%% Test de la fonction d'affichage pour train_data_2 

w = np.load("SVM-Noyaux/Hard_margin/w_2.npy", allow_pickle = True)
b = np.load("SVM-Noyaux/Hard_margin/b_2.npy", allow_pickle = True)

SVM.cam(w,b)

#%% Test de pourcentage de réussite des SVM (hard et soft) sur les deux bases de données 

# Création de la liste qui va contenir les matrices des images

Liste_faces = []

files = os.listdir("Test_data")
for i in range(1,len(files) + 1):
    Liste_faces.append(cv2.imread(f"Test_data/{i}.jpg",0).astype("float64"))
    Liste_faces[i-1] = cv2.resize(Liste_faces[i-1],(240,180))

taille_img_1 = 240
taille_img_2 = 180

train_data = np.zeros((taille_img_1*taille_img_2 + 2,len(files)))

for i in range(len(Liste_faces)-1):
    train_data[2:,i] = Liste_faces[i].reshape(-1, order = 'C')

for i in range(0,49):
    train_data[0,i] = 1
    
for i in range(50, len(Liste_faces)):
    train_data[0,i] = -1

train_data[1,0] = train_data[1,9] = train_data[1,11] = train_data[1,13] = train_data[1,15] = train_data[1,28] = train_data[1,40] = train_data[1,50] = train_data[1,51] = train_data[1,56] = train_data[1,60] = train_data[1,65] = train_data[1,73] = train_data[1,82] = train_data[1,85] = 11

np.save("SVM-Noyaux/Test_data.npy", train_data, allow_pickle = True)



# Test avec les vecteurs w et b de la première base de données en hard margin 

w = np.load("SVM-Noyaux/Hard_margin/w_2.npy", allow_pickle = True)
b = np.load("SVM-Noyaux/Hard_margin/b_2.npy", allow_pickle = True)

result = 0 
miss_train = 0
for i in range(np.shape(train_data)[1]):
    if train_data[0,i] == SVM.f(train_data[2:,i],w,b):
        result += 1 
    if train_data[0,i] != SVM.f(train_data[2:,i],w,b) and train_data[1,i] == 11:
        miss_train += 1

label = ["Mal catégorisé", "Bien catégorisé", "Donnée d'entraînement mal catégorisée"]
color = ["r", "g", "orange"]
wedges, texts, autotexts = plt.pie([np.shape(train_data)[1]-result-miss_train,result,miss_train], labels = label, colors = color, autopct='%1.1f%%', startangle=140)

for wedge in wedges:
    wedge.set_edgecolor('black')
    
plt.title("Performances de SVM hard margin entraîné sur train_data_2")

#%% Test en fonction du pas rho des performances des algorthmes pour hard margin train_data_1 

u,v = np.load("SVM-Noyaux/Hard_margin/u.npy", allow_pickle = True), np.load("SVM-Noyaux/Hard_margin/v.npy", allow_pickle = True) 

Rho = np.arange(0.5e-5,1e-8,-1e-8)

p = np.shape(u)[1]
q = np.shape(v)[1]

it = 15000 
tol = 1e-6

X = np.concatenate((-u.T,v.T), axis = 0)
ones = np.block([[np.ones((p,1))],[-np.ones((q,1))]])
C = np.concatenate((X,ones), axis = 1)

train_data = np.load("SVM-Noyaux/Hard_margin/test_data.npy", allow_pickle = True)

Result = []
It = []

for rho in Rho:
    mu, w, b, it = SVM.Hard_margin(rho, it, tol, u, v, C)
    result = 0 
    for i in range(np.shape(train_data)[1]):
        if train_data[0,i] == SVM.f(train_data[1:,i],w,b):
            result += 1 
        Result.append(result)
        It.append(it)

Result2 = Result[::100]
It2 = It[::100]

fig, ax = plt.subplots(1, 2, figsize=(17, 8))

ax[0].plot(Rho, Result, "xb")
ax[0].set_title(r"Performances de l'algorithme en fonction du pas $\rho$ sur test_data")
ax[1].plot(Rho, It, "xb")
ax[1].set_title(r"Itérations de l'algorithme en fonction du pas $\rho$ sur test_data")
plt.suptitle(r"Pour une tolérance $\varepsilon$ = 1e-6 et it = 15000 itérations.")
ax[0].set_xlabel(r"Valeurs de $\rho$")
ax[0].set_ylabel("Performance en pourcentage de l'algorithme")
ax[1].set_xlabel(r"Valeurs de $\rho$")
ax[1].set_ylabel("Nombre d'itérations")

#%% Test SVM avec la base de données BioID

# Lire un fichier PGM
image = Image.open("BioID/BioID_0001.pgm")

Images = [f for f in os.listdir("BioID") if f.endswith('.pgm')]
Imgs = []

for image in Images:
    Imgs.append(np.array(Image.open("BioID/" + image)).astype("float64").reshape(-1, order = "C")*fac + 0.01)
    
Liste_faces = []

files = os.listdir("Train_data_2")
for i in range(49,len(files) + 1):
    Liste_faces.append(cv2.imread(f"Train_data_2/{i}.jpg",0).astype("float64")*fac + 0.01)
    Liste_faces[i-49] = (cv2.resize(Liste_faces[i-49],(286,384))).reshape(-1, order = "C")   
    
taille_img = np.shape(Imgs[0])[0]

u = np.zeros((taille_img, len(Images)))
v = np.zeros((taille_img, len(Liste_faces)))

for i in range(len(Images)):
    u[:,i] = Imgs[i]
for i in range(len(Liste_faces)):
    v[:,i] = Liste_faces[i]
    
p = np.shape(u)[1]
q = np.shape(v)[1]

rho = 1e-10
it = 5000 
tol = 1e-6

X = np.concatenate((-u.T,v.T), axis = 0)
ones = np.block([[np.ones((p,1))],[-np.ones((q,1))]])
C = np.concatenate((X,ones), axis = 1)

mu, w, b = SVM.Hard_margin(rho, it, tol, u, v, C)
"""
#%% Test avec les bases de données de CIFAR pour v et FER-2013 pour u 
"""
with open('BD/cifar-10-batches-py/data_batch_1', 'rb') as f:
    train_cifar = pickle.load(f, encoding='bytes')

train_cifar_2 = list(train_cifar.values())[2]

imgs_v = list(np.zeros((250,32,32,3)))
imgs_v_bw = list(np.zeros((250,32,32)))
v = np.zeros((32*32,250))

for i in range(len(imgs_v)):
    for j in range(3):
        imgs_v[i][:,:,j] = train_cifar_2[i,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_v_bw[i] = cv2.cvtColor(imgs_v[i].astype("uint8"), cv2.COLOR_BGR2GRAY)
    v[:,i] = imgs_v_bw[i].reshape(-1,order = "C")*fac + 0.01
    
files = os.listdir('BD/train')
train_FER = []

for file in files:
    pict = os.listdir("BD/train/" + file)
    for picture in pict:
        train_FER.append(cv2.imread("BD/train/" + file + "/" + picture,0))

u = np.zeros((32*32, 250))
for i in range(250):
    u[:,i] = cv2.resize(train_FER[i], (32,32)).reshape(-1,order = 'C')*fac + 0.01
    
p = np.shape(u)[1]
q = np.shape(v)[1]

rho = 1e-7
it = 1000 
tol = 1e-6

X = np.concatenate((-u.T,v.T), axis = 0)
ones = np.block([[np.ones((p,1))],[-np.ones((q,1))]])
C = np.concatenate((X,ones), axis = 1)

#mu, w, b = SVM.Hard_margin(rho, it, tol, u, v, C)

#w = np.load("SVM-Noyaux/Hard_margin/w_BD.npy", allow_pickle = True)
#b = np.load("SVM-Noyaux/Hard_margin/b_BD.npy", allow_pickle = True)

#SVM.cam(w,b)

#%% Test avec les bases de données de CIFAR pour v et FER-2013 pour u en réalisant un whitening

with open('BD/cifar-10-batches-py/data_batch_1', 'rb') as f:
    train_cifar = pickle.load(f, encoding='bytes')

train_cifar_2 = list(train_cifar.values())[2]

imgs_v = list(np.zeros((10000,32,32,3)))
imgs_v_bw = list(np.zeros((10000,32,32)))
v = np.zeros((32*32,10000))

for i in range(len(imgs_v)):
    for j in range(3):
        imgs_v[i][:,:,j] = train_cifar_2[i,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_v_bw[i] = cv2.cvtColor(imgs_v[i].astype("uint8"), cv2.COLOR_BGR2GRAY)
    v[:,i] = imgs_v_bw[i].reshape(-1,order = "C")*fac + 0.01
    
files = os.listdir('BD/train')
train_FER = []

for file in files:
    pict = os.listdir("BD/train/" + file)
    for picture in pict:
        train_FER.append(cv2.imread("BD/train/" + file + "/" + picture,0))

u = np.zeros((32*32, len(train_FER)))
for i in range(len(train_FER)):
    u[:,i] = cv2.resize(train_FER[i], (32,32)).reshape(-1,order = 'C')*fac + 0.01
    
p = np.shape(u)[1]
q = np.shape(v)[1]

rho = 1e-7
it = 1000 
tol = 1e-6

X = np.concatenate((-u.T,v.T), axis = 0)
ones = np.block([[np.ones((p,1))],[-np.ones((q,1))]])
C = np.concatenate((X,ones), axis = 1)

unwhitened_data = np.zeros((np.shape(u)[0], np.shape(u)[1]+np.shape(v)[1]))
unwhitened_data[:,:p] = u
unwhitened_data[:,p:] = v

unwhitened_center_data = SVM.center(unwhitened_data)

whitened_data = SVM.whitening(unwhitened_center_data)

u_whitened = whitened_data[:,:p]
v_whitened = whitened_data[:,p:]

mu, w, b = SVM.Hard_margin(rho, it, tol, u_whitened, v_whitened, C)

SVM.cam(w,b)

#%% 

with open('BD/cifar-10-batches-py/data_batch_1', 'rb') as f:
    train_cifar = pickle.load(f, encoding='bytes')

train_cifar_2 = list(train_cifar.values())[2]

imgs_v = list(np.zeros((10000,32,32,3)))
imgs_v_bw = list(np.zeros((10000,32,32)))
v = np.zeros((32*32,10000))

for i in range(len(imgs_v)):
    for j in range(3):
        imgs_v[i][:,:,j] = train_cifar_2[i,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_v_bw[i] = cv2.cvtColor(imgs_v[i].astype("uint8"), cv2.COLOR_BGR2GRAY)
    v[:,i] = imgs_v_bw[i].reshape(-1,order = "C")
    
files = os.listdir('BD/train')
train_FER = []

for file in files:
    pict = os.listdir("BD/train/" + file)
    for picture in pict:
        train_FER.append(cv2.imread("BD/train/" + file + "/" + picture,0))

u = np.zeros((32*32, len(train_FER)))
for i in range(len(train_FER)):
    u[:,i] = cv2.resize(train_FER[i], (32,32)).reshape(-1,order = 'C')

p,q = np.shape(u)[1],np.shape(v)[1]

train_data = np.concatenate((u.T,v.T), axis = 0)
train_labels = np.block([[np.ones((p,1))],[-np.ones((q,1))]])
C = np.concatenate((train_data,train_labels), axis = 1)

m = np.shape(train_data)[0]

B = - np.ones((m,2))
for i in range(m):
    j = int(train_labels[i])
    B[i,j] = 1

# Initialisation de la matrice de Gramm
K = np.zeros((m,m))

# Remplissage de la matrice de Gramm
for i in range(m):
    for j in range(i,m):
        val = SVM.gauss(train_data[i,:],train_data[j,:])
        K[i,j] = val
        K[j,i] = val
    print(f"Remplissage de la matrice de Gramm : {(i/m)*100}%")

it = 10000
tol = 1e-6
rho = 1e-7

W,L = SVM.Moindre_carre_K(rho, it, tol, K, B)

#%% 

u,v = np.load("SVM-Noyaux/Hard_margin/u_2.npy", allow_pickle = True),np.load("SVM-Noyaux/Hard_margin/v_2.npy", allow_pickle = True)

p,q = np.shape(u)[1],np.shape(v)[1]

train_data = np.concatenate((u.T,v.T), axis = 0)
train_labels = np.block([[np.ones((p,1))],[-np.ones((q,1))]])
C = np.concatenate((train_data,train_labels), axis = 1)

m = np.shape(train_data)[0]

B = - np.ones((m,2))
for i in range(m):
    j = int(train_labels[i])
    B[i,j] = 1

# Initialisation de la matrice de Gramm
K = np.zeros((m,m))

# Remplissage de la matrice de Gramm
for i in range(m):
    for j in range(i,m):
        val = SVM.gauss(train_data[i,:],train_data[j,:])
        K[i,j] = val
        K[j,i] = val
    print(f"Remplissage de la matrice de Gramm : {(i/m)*100}%")
    

it = 1e5 
tol = 1e-6 
rho = 1e-7 
alpha = 1e-1

W,L = SVM.Moindre_carre_K(rho, it, tol, K, B, alpha)

#%% Test avec les bases de données de CIFAR pour v et FER-2013 pour u - SVM soft margin à noyaux SMO

with open('BD/cifar-10-batches-py/data_batch_1', 'rb') as f:
    train_cifar = pickle.load(f, encoding='bytes')

train_cifar_2 = list(train_cifar.values())[2]

imgs_v = list(np.zeros((250,32,32,3)))
imgs_v_bw = list(np.zeros((250,32,32)))
v = np.zeros((32*32,250))

for i in range(len(imgs_v)):
    for j in range(3):
        imgs_v[i][:,:,j] = train_cifar_2[i,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_v_bw[i] = cv2.cvtColor(imgs_v[i].astype("uint8"), cv2.COLOR_BGR2GRAY)
    v[:,i] = imgs_v_bw[i].reshape(-1,order = "C")*fac + 0.01
    
files = os.listdir('BD/train')
train_FER = []

for file in files:
    pict = os.listdir("BD/train/" + file)
    for picture in pict:
        train_FER.append(cv2.imread("BD/train/" + file + "/" + picture,0))

u = np.zeros((32*32, 250))
for i in range(250):
    u[:,i] = cv2.resize(train_FER[i], (32,32)).reshape(-1,order = 'C')*fac + 0.01
    
p = np.shape(u)[1]
q = np.shape(v)[1]

X = np.concatenate((u.T,v.T))
Y = np.zeros((500,1))
Y[:p,0] = 1
Y[p:,0] = -1 
alpha = 0.05
tol = 1e-3 
N = 500
start = t.time()
mu, b = SVM.SMO_simplified(alpha, tol, N, X, Y, SVM.gauss)
end = t.time()
temps_execution = end - start 
print(f"L'algorithme s'est résolu en {np.round(temps_execution,1)} secondes")

A = np.argwhere(mu[:,0] > 1e-15)
mu_new = np.zeros(np.shape(A))
Y_new = np.zeros(np.shape(A))
X_new = np.zeros((np.shape(A)[0],np.shape(X)[1]))
for i in range(np.shape(A)[0]):
    mu_new[i] = mu[A[i,0]]
    Y_new[i] = Y[A[i,0]]
    X_new[i,:] = X[A[i,0],:]  

Dy_new = np.diag(Y_new.reshape(-1,))
G_new = SVM.Gramm(SVM.gauss,X_new)

bon = 0 
est = []
for i in range(len(X_new)):
    x = X[i,:] 
    esti = SVM.f_D_data(x, mu_new, Dy_new, b, G_new[i,:].reshape(-1,1))
    est.append(esti)
    if np.sign(esti)*Y[i] > 0:
        bon += 1 
        
ratio = (bon/np.shape(A)[0])*100
"""
"""
Alpha = np.arange(0.01,1,0.01)
Ratio = np.zeros(np.shape(Alpha))
start = t.time()
for k in range(len(Alpha)):
    mu, b = SVM.SMO_simplified(Alpha[k], tol, N, X, Y, SVM.gauss)
    A = np.argwhere(mu[:,0] > 1e-15)
    mu_new = np.zeros(np.shape(A))
    Y_new = np.zeros(np.shape(A))
    X_new = np.zeros((np.shape(A)[0],np.shape(X)[1]))
    for i in range(np.shape(A)[0]):
        mu_new[i] = mu[A[i,0]]
        Y_new[i] = Y[A[i,0]]
        X_new[i,:] = X[A[i,0],:]
    
    Dy_new = np.diag(Y_new.reshape(-1,))
    G_new = SVM.Gramm(SVM.gauss,X_new)
    
    bon = 0 
    est = []
    for i in range(np.shape(X_new)[0]):
        x = X_new[i,:] 
        esti = SVM.f_D_data(x, mu_new, Dy_new, b, G_new[i,:].reshape(-1,1))
        est.append(esti)
        if np.sign(esti)*Y_new[i] > 0:
            bon += 1 
            
    Ratio[k] = (bon/np.shape(A)[0])*100 
    
end = t.time()
temps = np.round(end - start,1) 

plt.plot(Alpha,Ratio,"xr")
plt.title(r"Pourcentage de réussite en fonction de la marge $\alpha$")
plt.xlabel(r"Marge $\alpha$")
plt.ylabel("Pourcentage de réussite")

"""
#%% Tests sur la tolérance, N, alpha, et la taille des échantillons 
"""
with open('BD/cifar-10-batches-py/data_batch_1', 'rb') as f:
    train_cifar = pickle.load(f, encoding='bytes')

train_cifar_2 = list(train_cifar.values())[2]

imgs_v = list(np.zeros((250,32,32,3)))
imgs_v_bw = list(np.zeros((250,32,32)))
v = np.zeros((32*32,250))

for i in range(len(imgs_v)):
    for j in range(3):
        imgs_v[i][:,:,j] = train_cifar_2[i,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_v_bw[i] = cv2.cvtColor(imgs_v[i].astype("uint8"), cv2.COLOR_BGR2GRAY)
    v[:,i] = imgs_v_bw[i].reshape(-1,order = "C")*fac + 0.01
    
files = os.listdir('BD/train')
train_FER = []

for file in files:
    pict = os.listdir("BD/train/" + file)
    for picture in pict:
        train_FER.append(cv2.imread("BD/train/" + file + "/" + picture,0))

u = np.zeros((32*32, 250))
for i in range(250):
    u[:,i] = cv2.resize(train_FER[i], (32,32)).reshape(-1,order = 'C')*fac + 0.01
    
p = np.shape(u)[1]
q = np.shape(v)[1]

X = np.concatenate((u.T,v.T))
Y = np.zeros((500,1))
Y[:p,0] = 1
Y[p:,0] = -1 
alpha = 0.9
tol = 1e-3 
N = 500

Dy = np.diag(Y.reshape(-1,))
G = SVM.Gramm(SVM.gauss,X)

### Test sur la marge ###

Alpha = np.arange(0.01,1,0.01)
Ratio_Alpha = np.zeros(np.shape(Alpha))
Ratio_Alpha2 = np.zeros(np.shape(Alpha))
mu_Alpha = []
b_Alpha = []
temps_alpha = []
start = t.time()
for k in range(len(Alpha)):
    start2 = t.time()
    mu, b = SVM.SMO_simplified(Alpha[k], tol, N, X, Y, SVM.gauss)
    A = np.argwhere(mu[:,0] > 1e-15)
    mu_new = np.zeros(np.shape(A))
    mu_Alpha.append(mu_new)
    b_Alpha.append(b)
    Y_new = np.zeros(np.shape(A))
    X_new = np.zeros((np.shape(A)[0],np.shape(X)[1]))
    for i in range(np.shape(A)[0]):
        mu_new[i] = mu[A[i,0]]
        Y_new[i] = Y[A[i,0]]
        X_new[i,:] = X[A[i,0],:]
    
    Dy_new = np.diag(Y_new.reshape(-1,))
    G_new = SVM.Gramm(SVM.gauss,X_new)
    
    bon = 0 
    bon2 = 0
    for i in range(np.shape(X_new)[0]):
        x = X_new[i,:] 
        esti = SVM.f_D_data(x, mu_new, Dy_new, b, G_new[i,:].reshape(-1,1))
        if np.sign(esti)*Y_new[i] > 0:
            bon += 1 

    for i in range(np.shape(X)[0]):
        x = X[i,:]
        esti2 = SVM.f_D_data(x,mu,Dy,b,G[i,:].reshape(-1,1))
        if np.sign(esti2)*Y[i] > 0:
            bon2 +=1
            
    Ratio_Alpha[k] = (bon/np.shape(A)[0])*100 
    Ratio_Alpha2[k] = (bon2/np.shape(X)[0])*100
    
    end2 = t.time()
    temps_alpha.append(np.round(end2 - start2,1))
    
end = t.time()
Temps_Alpha = np.round(end - start,1) 

### Test sur la tolérance ###

Tol = np.arange(1e-2,1e-6,-0.05)
Ratio_tol = np.zeros(np.shape(Tol))
Ratio_tol2 = np.zeros(np.shape(Tol))
mu_tol = []
b_tol = []
temps_tol = [] 
start = t.time()
for k in range(len(Tol)):
    start2 = t.time()
    mu, b = SVM.SMO_simplified(alpha, Tol[k], N, X, Y, SVM.gauss)
    A = np.argwhere(mu[:,0] > 1e-15)
    mu_new = np.zeros(np.shape(A))
    mu_tol.append(mu_new)
    b_tol.append(b)
    Y_new = np.zeros(np.shape(A))
    X_new = np.zeros((np.shape(A)[0],np.shape(X)[1]))
    for i in range(np.shape(A)[0]):
        mu_new[i] = mu[A[i,0]]
        Y_new[i] = Y[A[i,0]]
        X_new[i,:] = X[A[i,0],:]
    
    Dy_new = np.diag(Y_new.reshape(-1,))
    G_new = SVM.Gramm(SVM.gauss,X_new)
    
    bon = 0 
    bon2 = 0
    for i in range(np.shape(X_new)[0]):
        x = X_new[i,:] 
        esti = SVM.f_D_data(x, mu_new, Dy_new, b, G_new[i,:].reshape(-1,1))
        if np.sign(esti)*Y_new[i] > 0:
            bon += 1 
    for i in range(np.shape(X)[0]):
        x = X[i,:]
        esti2 = SVM.f_D_data(x,mu,Dy,b,G[i,:].reshape(-1,1))
        if np.sign(esti2)*Y[i] > 0:
            bon2 +=1
            
    Ratio_tol[k] = (bon/np.shape(A)[0])*100 
    Ratio_tol2[k] = (bon2/np.shape(X)[0])*100
    
    end2 = t.time()
    temps_tol.append(np.round(end2 - start2,1))
    
end = t.time()
Temps_tol = np.round(end - start,1) 

### Test sur N ###

N_n = np.arange(100,2500,25)
Ratio_N = np.zeros(np.shape(N_n))
Ratio_N2 = np.zeros(np.shape(N_n))
mu_N = []
b_N = []
temps_N = []
start = t.time()
for k in range(len(N_n)):
    start2 = t.time()
    mu, b = SVM.SMO_simplified(alpha, tol, N_n[k], X, Y, SVM.gauss)
    A = np.argwhere(mu[:,0] > 1e-15)
    mu_new = np.zeros(np.shape(A))
    mu_N.append(mu_new)
    b_N.append(b)
    Y_new = np.zeros(np.shape(A))
    X_new = np.zeros((np.shape(A)[0],np.shape(X)[1]))
    for i in range(np.shape(A)[0]):
        mu_new[i] = mu[A[i,0]]
        Y_new[i] = Y[A[i,0]]
        X_new[i,:] = X[A[i,0],:]
    
    Dy_new = np.diag(Y_new.reshape(-1,))
    G_new = SVM.Gramm(SVM.gauss,X_new)
    
    bon = 0 
    bon2 = 0
    for i in range(np.shape(X_new)[0]):
        x = X_new[i,:] 
        esti = SVM.f_D_data(x, mu_new, Dy_new, b, G_new[i,:].reshape(-1,1))
        if np.sign(esti)*Y_new[i] > 0:
            bon += 1 
            
    for i in range(np.shape(X)[0]):
        x = X[i,:]
        esti2 = SVM.f_D_data(x,mu,Dy,b,G[i,:].reshape(-1,1))
        if np.sign(esti2)*Y[i] > 0:
            bon2 +=1
            
    Ratio_N[k] = (bon/np.shape(A)[0])*100 
    Ratio_N2[k] = (bon2/np.shape(X)[0])*100
    
    end2 = t.time()
    temps_N.append(np.round(end2-start2,1))
    
end = t.time()
Temps_N = np.round(end - start,1) 

### Test sur le nombre d'exemple d'entraînement ###

n_n = np.arange(250,10001,50)
Ratio_n = np.zeros(np.shape(n_n))
Ratio_n2 = np.zeros(np.shape(n_n))
mu_n = []
b_n = []
temps_n = []
start = t.time()
for k in range(len(n_n)):
    p = n_n[k]//2
    q = n_n[k]//2
    
    imgs_v = list(np.zeros((q,32,32,3)))
    imgs_v_bw = list(np.zeros((q,32,32)))
    v = np.zeros((32*32,q))

    for i in range(len(imgs_v)):
        for j in range(3):
            imgs_v[i][:,:,j] = train_cifar_2[i,(1024*j):(1024*(j+1))].reshape(32,32)
        imgs_v_bw[i] = cv2.cvtColor(imgs_v[i].astype("uint8"), cv2.COLOR_BGR2GRAY)
        v[:,i] = imgs_v_bw[i].reshape(-1,order = "C")*fac + 0.01
    
    u = np.zeros((32*32, p))
    for i in range(p):
        u[:,i] = cv2.resize(train_FER[i], (32,32)).reshape(-1,order = 'C')*fac + 0.01
        
    X = np.concatenate((u.T,v.T))
    Y = np.zeros((p+q,1))
    Y[:p,0] = 1 
    Y[p:,0] = -1
    
    Dy = np.diag(Y.reshape(-1,))
    G = SVM.Gramm(SVM.gauss,X)
    
    start2 = t.time()
    mu, b = SVM.SMO_simplified(alpha, tol, n_n[k], X, Y, SVM.gauss)
    A = np.argwhere(mu[:,0] > 1e-15)
    mu_new = np.zeros(np.shape(A))
    mu_n.append(mu_new)
    b_n.append(b)
    Y_new = np.zeros(np.shape(A))
    X_new = np.zeros((np.shape(A)[0],np.shape(X)[1]))
    for i in range(np.shape(A)[0]):
        mu_new[i] = mu[A[i,0]]
        Y_new[i] = Y[A[i,0]]
        X_new[i,:] = X[A[i,0],:]
    
    Dy_new = np.diag(Y_new.reshape(-1,))
    G_new = SVM.Gramm(SVM.gauss,X_new)
    
    bon = 0 
    bon2 = 0
    for i in range(np.shape(X_new)[0]):
        x = X_new[i,:] 
        esti = SVM.f_D_data(x, mu_new, Dy_new, b, G_new[i,:].reshape(-1,1))
        if np.sign(esti)*Y_new[i] > 0:
            bon += 1 
            
    for i in range(np.shape(X)[0]):
        x = X[i,:]
        esti2 = SVM.f_D_data(x,mu,Dy,b,G[i,:].reshape(-1,1))
        if np.sign(esti2)*Y[i] > 0:
            bon2 +=1
            
    Ratio_n[k] = (bon/np.shape(A)[0])*100 
    Ratio_n2[k] = (bon2/np.shape(X)[0])*100
    
    end2 = t.time()
    temps_n.append(np.round(end2-start2,1))
    
end = t.time()
Temps_n = np.round(end - start,1) 

fig,ax = plt.subplot((4,2),figsize = (15,7))
ax[0,0].plot(Alpha,Ratio_Alpha, "xr", label = r"Taux de réussite en fonction de $\alpha$")
ax[0,0].set_xlabel(r"Marge $\alpha$")
ax[0,0].set_ylabel("Taux de réussite (en %)")
ax[0,0].set_title(r"Taux de réussite en fonction de la marge $\alpha$")
ax[0,0].legend()

ax[0,1].plot(Alpha,np.array(temps_alpha), "xb", label = r"Temps d'exécution en fonction de $\alpha$")
ax[0,1].set_xlabel(r"Marge $\alpha$")
ax[0,1].set_ylabel("Temps d'éxecution (en secondes)")
ax[0,1].set_title(r"Temps d'éxecution de l'algorithme de SMO en fonction de $\alpha$")
ax[0,1].legend()


ax[1,0].plot(Tol,Ratio_tol, "xk", label = r"Taux de réussite en fonction de $\tau$")
ax[1,0].set_xlabel(r"Tolérance $\tau$")
ax[1,0].set_ylabel("Taux de réussite (en %)")
ax[1,0].set_title(r"Taux de réussite en fonction de la tolérance $\tau$")
ax[1,0].legend()

ax[1,1].plot(Tol,np.array(temps_tol), "xb", label = r"Temps d'exécution en fonction de $\tau$")
ax[1,1].set_xlabel(r"Tolérance $\tau$")
ax[1,1].set_ylabel("Temps d'éxecution (en secondes)")
ax[1,1].set_title(r"Temps d'éxecution de l'algorithme de SMO en fonction de $\tau$")
ax[1,1].legend()

ax[2,0].plot(N_n,Ratio_N, "xb", label = r"Taux de réussite en fonction de $N$")
ax[2,0].set_xlabel(r"Nombre $N$ de boucle sur les données sans modification de $\mu$")
ax[2,0].set_ylabel("Taux de réussite (en %)")
ax[2,0].set_title(r"Taux de réussite en fonction de $N$")
ax[2,0].legend()

ax[2,1].plot(N_n,np.array(temps_N), "xb", label = r"Temps d'exécution en fonction de $N$")
ax[2,1].set_xlabel(r"Nombre $N$ de boucle sur les données sans modification de $\mu$")
ax[2,1].set_ylabel("Temps d'éxecution (en secondes)")
ax[2,1].set_title(r"Temps d'éxecution de l'algorithme de SMO en fonction de $N$")
ax[2,1].legend()

ax[3,0].plot(n_n,Ratio_n, "xb", label = r"Taux de réussite en fonction de $n$")
ax[3,0].set_xlabel(r"Nombre $n$ d'exemple d'entraînement")
ax[3,0].set_ylabel("Taux de réussite (en %)")
ax[3,0].set_title(r"Taux de réussite en fonction de $n$")
ax[3,0].legend()

ax[3,1].plot(N_n,np.array(temps_N), "xb", label = r"Temps d'exécution en fonction de $n$")
ax[3,1].set_xlabel(r"Nombre $n$ d'exemple d'entraînement")
ax[3,1].set_ylabel("Temps d'éxecution (en secondes)")
ax[3,1].set_title(r"Temps d'éxecution de l'algorithme de SMO en fonction de $n$")
ax[3,1].legend()
"""
#%% Test avec les bases de données de CIFAR pour v et FER-2013 pour u - SVM soft margin à noyaux SMO avec la class SVM
"""
with open('BD/cifar-10-batches-py/data_batch_1', 'rb') as f:
    train_cifar = pickle.load(f, encoding='bytes')

train_cifar_2 = list(train_cifar.values())[2]

files = os.listdir('BD/train')
train_FER = []

for file in files:
    pict = os.listdir("BD/train/" + file)
    for picture in pict:
        train_FER.append(cv2.imread("BD/train/" + file + "/" + picture,0))

n = 4000

imgs_v = list(np.zeros((n//2,32,32,3)))
imgs_v_bw = list(np.zeros((n//2,32,32)))
v = np.zeros((32*32,n//2))

for i in range(len(imgs_v)):
    for j in range(3):
        imgs_v[i][:,:,j] = train_cifar_2[i,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_v_bw[i] = cv2.cvtColor(imgs_v[i].astype("uint8"), cv2.COLOR_BGR2GRAY)
    v[:,i] = imgs_v_bw[i].reshape(-1,order = "C")*fac + 0.01

u = np.zeros((32*32, n//2))
for i in range(n//2):
    u[:,i] = cv2.resize(train_FER[i], (32,32)).reshape(-1,order = 'C')*fac + 0.01
    
p = np.shape(u)[1]
q = np.shape(v)[1]

X = np.concatenate((u.T,v.T))
Y = np.zeros((n,1))
Y[:p,0] = 1
Y[p:,0] = -1 

### Parameters of the class ### 

alpha = 0.9
tol = 1e-3 
eps = 1e-3
mu = np.zeros((np.shape(X)[0],1))
b = 0 

model = SVMC.SMOModel(X, Y, alpha, SVM.gauss, mu, b, np.zeros(np.shape(X)[0]), eps, tol)

K = SVM.Gramm(model.kernel,X)

for i in range(len(X)):
    initial_error = SVMC.decision_function2(model.mu, model.y, model.X, model.X[i,:], model.b) - model.y[i]
    model.errors[i] = initial_error

output = SVMC.train(model,K)

precision, recall, F1 = SVM.metric(model.y,model.mu,model.b,K)
print(precision, recall, F1)

SVM.Affiche_acc(model.y, model.mu, model.b, K)

SVM.cam(model.mu,model.b,SVM.gauss,model.X,model.y)
"""

#%% Test avec la base de données CIFAR-100 avec Uzawa

with open("BD/cifar-100-python/train", 'rb') as f:
    train_cifar_100 = pickle.load(f,encoding = "bytes")

# data1_norm = (data_float - np.min(data_float)) / (np.max(data_float) - np.min(data_float)) # data float est juste une conversion pour mettre en float, pas obligatoire

n = 2000

train_cifar = list(train_cifar_100.values())
train_cifar_data = list(train_cifar_100.values())[4]
indices_visages = np.argwhere((np.array(train_cifar[2]) == 98) | (np.array(train_cifar[2]) == 46))
indices_non_visages = np.argwhere(np.array(train_cifar[2]) != indices_visages)[:,1]

imgs_v = list(np.zeros((n//2,32,32,3)))
imgs_v_bw = list(np.zeros((n//2,32,32)))
v = np.zeros((n//2,32*32))

imgs_u = list(np.zeros((n//2,32,32,3)))
imgs_u_bw = list(np.zeros((n//2,32,32)))
u = np.zeros((n//2,32*32))
 
compt = 0
while compt < n//2:
    u_ind = np.random.choice(indices_visages[:,0])
    v_ind = np.random.choice(indices_non_visages)
    for j in range(3):
        imgs_v[compt][:,:,j] = train_cifar_data[v_ind,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_v_bw[compt] = cv2.cvtColor(imgs_v[compt].astype("uint8"), cv2.COLOR_BGR2GRAY)
    v[compt,:] = cv2.normalize(imgs_v_bw[compt].reshape(-1,order = "C").astype("float64"), None, 0,1,cv2.NORM_MINMAX).reshape(-1,)
    
    for j in range(3):
        imgs_u[compt][:,:,j] = train_cifar_data[u_ind,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_u_bw[compt] = cv2.cvtColor(imgs_u[compt].astype("uint8"), cv2.COLOR_BGR2GRAY)
    u[compt,:] = cv2.normalize(imgs_u_bw[compt].reshape(-1,order = "C").astype("float64"), None, 0,1,cv2.NORM_MINMAX).reshape(-1,)
    
    compt += 1 

u_reshape = SVM.crop_and_resize(imgs_u[0], 32*5, 32*5).astype("uint8")
cv2.imshow('test',u_reshape)

# Charger le modèle de super-résolution
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("EDSR_x4.pb")  # Charger un modèle pré-entraîné (ici x4)
sr.setModel("edsr", 4)  # Spécifier le modèle et le facteur d'agrandissement

# Appliquer la super-résolution
sr_img = sr.upsample(u_reshape)

# Sauvegarde ou affichage
cv2.imshow("Super Resolution", sr_img)

p = np.shape(u)[0]
q = np.shape(v)[0]

X = np.concatenate((u,v))
K = SVM.Gramm(SVM.gauss, X)

y = np.zeros((n,1))
y[:p,0] = 1
y[p:,0] = -1

Ku = K[:,:p]
Kv = K[:,p:]
k = np.concatenate((-Ku.T,Kv.T), axis = 0)
ones = np.block([[np.ones((p,1))],[-np.ones((q,1))]])
C = np.concatenate((k,ones), axis = 1)

alpha = 0.5
rho = 1e-7
it = 15000
tol = 1e-7

mu,a,b = SVM.Soft_margin_noyaux(alpha, rho, it, tol, u, v, C)

accuracy, precision, recall, F1 = SVM.metric_uza(y, a, b, X, SVM.gauss,X)
print(accuracy, precision, recall, F1)

#%% Test Uzawa noyaux sur mnist 

data = np.genfromtxt("BD/mnist_test.csv", delimiter = ",")

ind_zero = np.argwhere(data[:,0] == 0)
ind_nonzero = np.argwhere(data[:,0] != 0)
u = data[ind_zero.reshape(-1,),1:]*fac + 0.01
v = data[ind_nonzero.reshape(-1,),1:]*fac + 0.01

p = np.shape(u)[0]
q = np.shape(v)[0]

X = np.concatenate((u,v))
K = SVM.Gramm(SVM.gauss, X)

y = np.zeros((np.shape(X)[0],1))
y[:p,0] = 1
y[p:,0] = -1

Ku = K[:,:p]
Kv = K[:,p:]
k = np.concatenate((-Ku.T,Kv.T), axis = 0)
ones = np.block([[np.ones((p,1))],[-np.ones((q,1))]])
C = np.concatenate((k,ones), axis = 1)

alpha = 0.00001
rho = 1e-7
it = 1000
tol = 1e-8

mu,a,b = SVM.Soft_margin_noyaux(alpha, rho, it, tol, u, v, C)

accuracy, precision, recall, F1,confusion_matrix = SVM.metric_uza(y, a, b, X, SVM.gauss,X)
print(accuracy, precision, recall, F1)

print(SVM.fD_uza(a, b, X, SVM.gauss, u[159,:], y))

### Avec SMO ### 

tol = 1e-4
eps = 1e-4
mu = np.zeros((np.shape(X)[0],1))
b = 0 

model = SVMC.SMOModel(X, y, alpha, SVM.gauss, mu, b, np.zeros(np.shape(X)[0]), eps, tol)

K = SVM.Gramm(model.kernel,X)

for i in range(len(X)):
    initial_error = SVMC.decision_function2(model.mu, model.y, model.X, model.X[i,:], model.b) - model.y[i]
    model.errors[i] = initial_error

output = SVMC.train(model,K)

accuracy, precision, recall, F1, confusion_matrix = SVM.metric(model.y,model.mu,model.b,K)
print(accuracy, precision, recall, F1)

#%% Test SMO noyaux sur iris 

#%% Test avec test 

with open("BD/cifar-100-python/test", 'rb') as f:
    test_cifar_100 = pickle.load(f,encoding = "bytes")
    
test_cifar = list(test_cifar_100.values())
test_cifar_data = test_cifar[4]
indices_visages = np.argwhere((np.array(test_cifar[2]) == 98) | (np.array(test_cifar[2]) == 46))
indices_non_visages =  np.argwhere(np.array(test_cifar[2]) != indices_visages)[:,1]

n_tilde = 400

imgs_v_tilde = list(np.zeros((n_tilde//2,32,32,3)))
imgs_v_tilde_bw = list(np.zeros((n_tilde//2,32,32)))
v_tilde = np.zeros((n_tilde//2,32*32))

imgs_u_tilde = list(np.zeros((n_tilde//2,32,32,3)))
imgs_u_tilde_bw = list(np.zeros((n_tilde//2,32,32)))
u_tilde = np.zeros((n_tilde//2,32*32))
 
compt = 0
while compt < n_tilde//2:
    u_ind = indices_visages[compt]
    v_ind = np.random.choice(indices_non_visages)
    for j in range(3):
        imgs_v[compt][:,:,j] = test_cifar_data[v_ind,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_v_bw[compt] = cv2.cvtColor(imgs_v[compt].astype("uint8"), cv2.COLOR_BGR2GRAY)
    v[compt,:] = cv2.normalize(imgs_v_bw[compt].reshape(-1,order = "C").astype("float64"), None, 0,1,cv2.NORM_MINMAX).reshape(-1,)
    
    for j in range(3):
        imgs_u_tilde[compt][:,:,j] = test_cifar_data[u_ind,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_u_tilde_bw[compt] = cv2.cvtColor(imgs_u_tilde[compt].astype("uint8"), cv2.COLOR_BGR2GRAY)
    u_tilde[compt,:] = cv2.normalize(imgs_u_tilde_bw[compt].reshape(-1,order = "C").astype("float64"), None, 0,1,cv2.NORM_MINMAX).reshape(-1,)
    
    compt += 1 

y = np.ones((n_tilde,1))
y[n_tilde//2:,0] = -1

accuracy, precision, recall, F1 = SVM.metric_uza(y, a, b, X, SVM.gauss,np.concatenate((u_tilde,v_tilde)))
print(accuracy, precision, recall, F1)

SVM.cam(a,b[0],SVM.gauss,X,y)

#%% Test avec la base de données CIFAR-100 

with open("BD/cifar-100-python/train", 'rb') as f:
    train_cifar_100 = pickle.load(f,encoding = "bytes")

n = 2000

train_cifar = list(train_cifar_100.values())
train_cifar_data = list(train_cifar_100.values())[4]
indices_visages = np.argwhere((np.array(train_cifar[2]) == 98) | (np.array(train_cifar[2]) == 46))
indices_non_visages = np.argwhere(np.array(train_cifar[2]) != indices_visages)[:,1]

imgs_v = list(np.zeros((n//2,32,32,3)))
imgs_v_bw = list(np.zeros((n//2,32,32)))
v = np.zeros((n//2,32*32))

imgs_u = list(np.zeros((n//2,32,32,3)))
imgs_u_bw = list(np.zeros((n//2,32,32)))
u = np.zeros((n//2,32*32))
 
compt = 0
while compt < n//2:
    u_ind = np.random.choice(indices_visages[:,0])
    v_ind = np.random.choice(indices_non_visages)
    for j in range(3):
        imgs_v[compt][:,:,j] = train_cifar_data[v_ind,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_v_bw[compt] = cv2.cvtColor(imgs_v[compt].astype("uint8"), cv2.COLOR_BGR2GRAY)
    v[compt,:] = cv2.normalize(imgs_v_bw[compt].reshape(-1,order = "C").astype("float64"), None, 0,1,cv2.NORM_MINMAX).reshape(-1,)
    
    for j in range(3):
        imgs_u[compt][:,:,j] = train_cifar_data[u_ind,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_u_bw[compt] = cv2.cvtColor(imgs_u[compt].astype("uint8"), cv2.COLOR_BGR2GRAY)
    u[compt,:] = cv2.normalize(imgs_u_bw[compt].reshape(-1,order = "C").astype("float64"), None, 0,1,cv2.NORM_MINMAX).reshape(-1,)
    
    compt += 1 

p = np.shape(u)[0]
q = np.shape(v)[0]

X = np.concatenate((u,v))
Y = np.zeros((n,1))
Y[:p,0] = 1
Y[p:,0] = -1 

### Paramètres de la classe ### 

alpha = 0.8
tol = 1e-3 
eps = 1e-3
mu = np.zeros((np.shape(X)[0],1))
b = 0 

model = SVMC.SMOModel(X, Y, alpha, SVM.gauss, mu, b, np.zeros(np.shape(X)[0]), eps, tol)

K = SVM.Gramm(model.kernel,X)

for i in range(len(X)):
    initial_error = SVMC.decision_function2(model.mu, model.y, model.X, model.X[i,:], model.b) - model.y[i]
    model.errors[i] = initial_error

output = SVMC.train(model,K)

SVM.Affiche_acc(model.y, model.mu, model.b, K)

precision, recall, F1 = SVM.metric(model.y,model.mu,model.b,K)
print(precision, recall, F1)

SVM.cam(model.mu,model.b,SVM.gauss,model.X,model.y)

#%% Test de notre SMO pour SVM à marge souple en fonction de la marge, la tolérance, le nombre d'échantillon et l'hyper-paramètre du noyau gaussien 

### Chargement des données ###
debut = t.time()
with open('BD/cifar-10-batches-py/data_batch_1', 'rb') as f:
    train_cifar = pickle.load(f, encoding='bytes')

train_cifar_2 = list(train_cifar.values())[2]

files = os.listdir('BD/train')
train_FER = []

for file in files:
    if file == 'neutral':
        pict = os.listdir("BD/train/" + file)
        for picture in pict:
            train_FER.append(cv2.imread("BD/train/" + file + "/" + picture,0))

imgs_v = list(np.zeros((500,32,32,3)))
imgs_v_bw = list(np.zeros((500,32,32)))
v = np.zeros((32*32,500))

for i in range(len(imgs_v)):
    for j in range(3):
        imgs_v[i][:,:,j] = train_cifar_2[i,(1024*j):(1024*(j+1))].reshape(32,32)
    imgs_v_bw[i] = cv2.cvtColor(imgs_v[i].astype("uint8"), cv2.COLOR_BGR2GRAY)
    v[:,i] = imgs_v_bw[i].reshape(-1,order = "C")*fac + 0.01

u = np.zeros((32*32, 500))
for i in range(500):
    u[:,i] = cv2.resize(train_FER[i], (32,32)).reshape(-1,order = 'C')*fac + 0.01
    
p = np.shape(u)[1]
q = np.shape(v)[1]

X = np.concatenate((u.T,v.T))
Y = np.zeros((1000,1))
Y[:p,0] = 1
Y[p:,0] = -1 
fin = t.time()
duree1 = fin - debut

#%% 

### Paramètres généraux de la classe ### 

alpha = 0.9
tol = 1e-3 
eps = 1e-3
mu_init = np.zeros((np.shape(X)[0],1))
b_init = 0 
K = SVM.Gramm(SVM.gauss,X)

### Vecteurs des paramètres sur lesquels on boucle ### 

Alpha = np.arange(0.01,1,0.01)
Tol = np.logspace(-2, -6, num=10)
n_n = np.arange(250,2500,100)
Sigma = np.arange(10,20,0.5)

### Listes des métriques et temps en fonction des paramètres ### 

Precision_Alpha = []
Recall_Alpha = []
F1_Alpha = []
temps_alpha = []

Precision_tol = []
Recall_tol = []
F1_tol = []
temps_tol = []

Precision_n = []
Recall_n = []
F1_n = []
temps_n = []

Precision_Sigma = []
Recall_Sigma = []
F1_Sigma = []
temps_Sigma = []

### Boucle sur la marge ### 

for i in range(len(Alpha)):
    model = SVMC.SMOModel(X, Y, Alpha[i], SVM.gauss, mu_init, b_init, np.zeros(np.shape(X)[0]), eps, tol)
    for j in range(len(X)):
        initial_error = SVMC.decision_function2(model.mu, model.y, model.X, model.X[j,:], model.b) - model.y[j][0]
        model.errors[j] = initial_error
    
    start = t.time()
    model = SVMC.train(model,K)
    end = t.time()
    temps_alpha.append(np.round(end - start,2))

    precision, recall, F1 = SVM.metric(model.y,model.mu,model.b,K)
    Precision_Alpha.append(precision*100)
    Recall_Alpha.append(recall*100)
    F1_Alpha.append(F1*100)
    avancement = np.round(((i+1)/len(Alpha))*100,2)
    print(f"Complété pour la marge à {avancement} %")

### Boucle sur la tolérance ### 

for i in range(len(Tol)):
    model = SVMC.SMOModel(X, Y, alpha, SVM.gauss, mu_init, b_init, np.zeros(np.shape(X)[0]), eps, Tol[i])
    for j in range(len(X)):
        initial_error = SVMC.decision_function2(model.mu, model.y, model.X, model.X[j,:], model.b) - model.y[j][0]
        model.errors[j] = initial_error
    
    start = t.time()
    model = SVMC.train(model,K)
    end = t.time()
    temps_tol.append(np.round(end - start,2))

    precision, recall, F1 = SVM.metric(model.y,model.mu,model.b,K)
    Precision_tol.append(precision*100)
    Recall_tol.append(recall*100)
    F1_tol.append(F1*100)
    avancement = np.round(((i+1)/len(Tol))*100,2)
    print(f"Complété pour la tolérance à {avancement} %")

### Boucle sur l'hyper paramètre Sigma ### 

for i in range(len(Sigma)):
    SVM.sig = Sigma[i]
    model = SVMC.SMOModel(X, Y, alpha, SVM.gauss, mu_init, b_init, np.zeros(np.shape(X)[0]), eps, tol)
    for j in range(len(X)):
        initial_error = SVMC.decision_function2(model.mu, model.y, model.X, model.X[j,:], model.b) - model.y[j][0]
        model.errors[j] = initial_error
    
    K = SVM.Gramm(SVM.gauss,X)
    
    start = t.time()
    model = SVMC.train(model,K)
    end = t.time()
    temps_Sigma.append(np.round(end - start,2))

    precision, recall, F1 = SVM.metric(model.y,model.mu,model.b,K)
    Precision_Sigma.append(precision*100)
    Recall_Sigma.append(recall*100)
    F1_Sigma.append(F1*100)
    avancement = np.round(((i+1)/len(Sigma))*100,2)
    print(f"Complété pour sigma à {avancement} %")

### Boucle sur le nombre d'échantillon ### 

SVM.sig = 10 
for i in range(len(n_n)):

    imgs_v = list(np.zeros((n_n[i]//2,32,32,3)))
    imgs_v_bw = list(np.zeros((n_n[i]//2,32,32)))
    v = np.zeros((32*32,n_n[i]//2))

    for k in range(len(imgs_v)):
        for j in range(3):
            imgs_v[k][:,:,j] = train_cifar_2[k,(1024*j):(1024*(j+1))].reshape(32,32)
        imgs_v_bw[k] = cv2.cvtColor(imgs_v[k].astype("uint8"), cv2.COLOR_BGR2GRAY)
        v[:,k] = imgs_v_bw[k].reshape(-1,order = "C")*fac + 0.01

    u = np.zeros((32*32, n_n[i]//2))
    for k in range(n_n[i]//2):
        u[:,k] = cv2.resize(train_FER[k], (32,32)).reshape(-1,order = 'C')*fac + 0.01
        
    p = np.shape(u)[1]
    q = np.shape(v)[1]

    X = np.concatenate((u.T,v.T))
    Y = np.zeros((n_n[i],1))
    Y[:p,0] = 1
    Y[p:,0] = -1 
    K = SVM.Gramm(SVM.gauss,X)
    mu_init = np.zeros((np.shape(X)[0],1))
    b_init = 0
    
    model = SVMC.SMOModel(X, Y, alpha, SVM.gauss, mu_init, b_init, np.zeros(np.shape(X)[0]), eps, tol)
    for j in range(len(X)):
        initial_error = SVMC.decision_function2(model.mu, model.y, model.X, model.X[j,:], model.b) - model.y[j][0]
        model.errors[j] = initial_error
    
    start = t.time()
    model = SVMC.train(model,K)
    end = t.time()
    temps_n.append(np.round(end - start,2))

    precision, recall, F1 = SVM.metric(model.y,model.mu,model.b,K)
    Precision_n.append(precision*100)
    Recall_n.append(recall*100)
    F1_n.append(F1*100)
    avancement = np.round(((i+1)/len(n_n))*100,2)
    print(f"Complété pour n à {avancement} %")

### Plots des résultats ### 

n_n = n_n[:len(temps_n)]
fig,ax = plt.subplots(4,2,figsize = (21,12))

ax[0,0].plot(Alpha,np.array(Precision_Alpha), "xr", label = r"Precision en fonction de $\alpha$")
ax[0,0].plot(Alpha,np.array(Recall_Alpha), "xb", label = r"Recall en fonction de $\alpha$")
ax[0,0].plot(Alpha,np.array(F1_Alpha), "xg", label = r"F1 en fonction de $\alpha$")
ax[0,0].set_xlabel(r"Marge $\alpha$")
ax[0,0].set_ylabel("Métrique (en %)")
ax[0,0].set_title(r"Métriques en fonction de la marge $\alpha$")
ax[0,0].legend()

ax[0,1].plot(Alpha,np.array(temps_alpha), "xb", label = r"Temps d'exécution en fonction de $\alpha$")
ax[0,1].set_xlabel(r"Marge $\alpha$")
ax[0,1].set_ylabel("Temps d'éxecution (en secondes)")
ax[0,1].set_title(r"Temps d'éxecution de l'algorithme de SMO en fonction de $\alpha$")
ax[0,1].legend()


ax[1,0].plot(Tol,np.array(Precision_tol), "xr", label = r"Precision en fonction de $\tau$")
ax[1,0].plot(Tol,np.array(Recall_tol), "xb", label = r"Recall en fonction de $\tau$")
ax[1,0].plot(Tol,np.array(F1_tol), "xg", label = r"F1 en fonction de $\tau$")
ax[1,0].set_xlabel(r"Tolérance $\tau$")
ax[1,0].set_ylabel("Métrique (en %)")
ax[1,0].set_title(r"Métriques en fonction de la tolérance $\tau$")
ax[1,0].legend()

ax[1,1].plot(Tol,np.array(temps_tol), "xb", label = r"Temps d'exécution en fonction de $\tau$")
ax[1,1].set_xlabel(r"Tolérance $\tau$")
ax[1,1].set_ylabel("Temps d'éxecution (en secondes)")
ax[1,1].set_title(r"Temps d'éxecution de l'algorithme de SMO en fonction de $\tau$")
ax[1,1].legend()


ax[2,0].plot(n_n,Precision_n, "xr", label = r"Precision en fonction de $n$")
ax[2,0].plot(n_n,Recall_n, "xb", label = r"Recall en fonction de $n$")
ax[2,0].plot(n_n,F1_n, "xg", label = r"F1 en fonction de $n$")
ax[2,0].set_xlabel(r"Taille de l'échantillon d'entraînement $n$")
ax[2,0].set_ylabel("Métrique (en %)")
ax[2,0].set_title(r"Métriques en fonction de $n$")
ax[2,0].legend()

ax[2,1].plot(n_n,np.array(temps_n), "xb", label = r"Temps d'exécution en fonction de $n$")
ax[2,1].set_xlabel(r"Taille de l'échantillon d'entraînement $n$")
ax[2,1].set_ylabel("Temps d'éxecution (en secondes)")
ax[2,1].set_title(r"Temps d'éxecution de l'algorithme de SMO en fonction de $n$")
ax[2,1].legend()


ax[3,0].plot(Sigma,np.array(Precision_Sigma), "xr", label = r"Precision en fonction de $\sigma$")
ax[3,0].plot(Sigma,np.array(Recall_Sigma), "xb", label = r"Recall en fonction de $\sigma$")
ax[3,0].plot(Sigma,np.array(F1_Sigma), "xg", label = r"F1 en fonction de $\sigma$")
ax[3,0].set_xlabel(r"Hyper paramètre $\sigma$ du noyau gaussien")
ax[3,0].set_ylabel("Métrique (en %)")
ax[3,0].set_title(r"Métriques en fonction de $\sigma$")
ax[3,0].legend()

ax[3,1].plot(Sigma,np.array(temps_Sigma), "xb", label = r"Temps d'exécution en fonction de $\sigma$")
ax[3,1].set_xlabel(r"Hyper paramètre $\sigma$ du noyau gaussien")
ax[3,1].set_ylabel("Temps d'éxecution (en secondes)")
ax[3,1].set_title(r"Temps d'éxecution de l'algorithme de SMO en fonction de $\sigma$")
ax[3,1].legend()

#%% Nouveau test avec les features de Haar basés sur la variance 

### Chargement des données ###

with open('BD/cifar-10-batches-py/data_batch_1', 'rb') as f:
    train_cifar = pickle.load(f, encoding='bytes')

train_cifar_2 = list(train_cifar.values())[2]

v = np.zeros((928,500))
for i in range(500):
    v_RGB = np.zeros((32,32,3))
    for j in range(3):
        v_RGB[:,:,j] = train_cifar_2[i,(1024*j):(1024*(j+1))].reshape(32,32)
    v_bw = cv2.cvtColor(v_RGB.astype("uint8"), cv2.COLOR_BGR2GRAY)
    Features = SVM.Haar_features(cv2.resize(v_bw,(64,64)))
    v[:,i] = Features

files = os.listdir('BD/train')
train_FER = []

for file in files:
    pict = os.listdir("BD/train/" + file)
    for picture in pict:
        train_FER.append(cv2.imread("BD/train/" + file + "/" + picture,0))

train_FER.pop(2)

u = np.zeros((928, 500))
for i in range(500):
    Features = SVM.Haar_features(cv2.resize(train_FER[i], (64,64)))
    u[:,i] = Features 
    
p = np.shape(u)[1]
q = np.shape(v)[1]

X = np.concatenate((u.T,v.T))
Y = np.zeros((1000,1))
Y[:p,0] = 1
Y[p:,0] = -1 
sig = 1.2
K = SVM.Gramm(SVM.gauss,X,sig)
K = SVM.Gramm(SVM.kern_poly,X,(1.5,3))

alpha = 0.1
tol = 1e-3 
eps = 1e-2
mu_init = np.zeros((np.shape(X)[0],1))
b_init = 0

# model = SVMC.SMOModel(X, Y, alpha, SVM.kern_poly, mu_init, b_init, np.zeros(np.shape(X)[0]), eps, tol)
model = SVMC.SMOModel(X, Y, alpha, SVMC.gaussian_kernel, mu_init, b_init, np.zeros(np.shape(X)[0]), eps, tol)
for j in range(len(X)):
    # initial_error = SVMC.decision_function2(model.mu, model.y, model.X, model.X[j,:], model.b, SVM.kern_poly, (1.5,3)) - model.y[j][0]
    initial_error = SVMC.decision_function2(model.mu, model.y, model.X, model.X[j,:], model.b, SVMC.gaussian_kernel, sig) - model.y[j][0]
    model.errors[j] = initial_error

start = t.time()
model = SVMC.train(model, K, sig)
end = t.time()
temps = np.round(end - start,2)

accuracy, precision, recall, F1,confusion_matrix = SVM.metric(model.y,model.mu,model.b,K)
print("Accuracy = ", np.round(accuracy*100,2))
print("Precision = ", np.round(precision*100,2))
print("Recall = ", np.round(recall*100,2))
print("F1 = ", np.round(F1*100,2))

#%% 

import importlib 
importlib.reload(SVMC)

#%% Test de la fonction routine 

debut2 = t.time()
X_routine,Y_routine = SVM.routine_test('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 500, 500)
fin2 = t.time()
duree2 = fin2 - debut2 

X_test,Y_test = SVM.routine_test("BD/train", "BD/cifar-10-batches-py/data_batch_1", 1000, 1000)
X_test = X_test[750:1750]
Y_test = Y_test[750:1750]

#%% Test de la modification sur SVM_class 

X,Y = SVM.routine_test('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 500, 500)

alpha = 0.79
tol = 1e-3 
eps = 1e-2
mu_init = np.zeros(np.shape(X)[0])
b_init = 0
error_init = np.zeros(1000)

sig = 2500
K = SVM.Gramm(SVM.gauss,X,sig)

#%% 

model = SVMC.SMOModel(X = X, y = Y, alpha =  alpha, kernel = SVMC.gaussian_kernel, mu = mu_init, b = b_init,
                      errors = error_init, eps = eps, tol = tol, args = sig)

initial_errors = SVMC.decision_function(model, X) - model.y
model.errors = initial_errors

model = SVMC.train(model)

accuracy, precision, recall, F1,confusion_matrix = SVM.metric(model, model.X)
print("Accuracy = ", np.round(accuracy*100,2))
print("Precision = ", np.round(precision*100,2))
print("Recall = ", np.round(recall*100,2))
print("F1 = ", np.round(F1*100,2))

accuracy_test, precision_test, recall_test, F1_test,confusion_matrix_test = SVM.metric(model, X_test)
print("Accuracy test = ", np.round(accuracy_test*100,2))
print("Precision test = ", np.round(precision_test*100,2))
print("Recall test = ", np.round(recall_test*100,2))
print("F1 test = ", np.round(F1_test*100,2))

#%% 

start = t.time()
model = SVMC.train(model, K, sig)
end = t.time()
temps = np.round(end - start,2)

accuracy, precision, recall, F1,confusion_matrix = SVM.metric(model.y,model.mu,model.b,K)
print("Accuracy = ", np.round(accuracy*100,2))
print("Precision = ", np.round(precision*100,2))
print("Recall = ", np.round(recall*100,2))
print("F1 = ", np.round(F1*100,2))

#%% Adaptation des fonctions de metric pour des données tests non présentes dans les données d'entraînement 

accuracy_test, precision_test, recall_test, F1_test,confusion_matrix_test = SVM.metric_test(X_test,X,model.mu,model.y,SVM.gauss,model.b)
print("Accuracy = ", np.round(accuracy_test*100,2))
print("Precision = ", np.round(precision_test*100,2))
print("Recall = ", np.round(recall_test*100,2))
print("F1 = ", np.round(F1_test*100,2))

#%% 

SVM.cam(model.mu,model.b,SVM.gauss,model.X,model.y)

#%% Test de l'algo sur un exemple du site : c'est validé ! 

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def plot_decision_boundary(model, ax, sig, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
        """Plots the model's decision boundary on the input axes object.
        Range of decision boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grid`, `ax`)."""
        
        # Generate coordinate grid of shape [resolution x resolution]
        # and evaluate the model over the entire space
        xrange = np.linspace(model.X[:,0].min(), model.X[:,0].max(), resolution)
        yrange = np.linspace(model.X[:,1].min(), model.X[:,1].max(), resolution)
        grid = [[SVMC.decision_function(model, np.array([xr, yr])) for xr in xrange] for yr in yrange]
        grid = np.array(grid).reshape(len(xrange), len(yrange))
        
        # Plot decision contours using grid and
        # make a scatter plot of training data
        ax.contour(xrange, yrange, grid, levels=levels, linewidths=(1, 1, 1),
                   linestyles=('--', '-', '--'), colors=colors)
        ax.scatter(model.X[:,0], model.X[:,1],
                   c=model.y, cmap=plt.cm.viridis, lw=0, alpha=0.25)
        
        # Plot support vectors (non-zero alphas)
        # as circled points (linewidth > 0)
        mask = (np.round(model.mu, decimals=2) != 0.0).flatten()
        ax.scatter(model.X[mask,0], model.X[mask,1],
                   c=model.y[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')
        
        return grid, ax

X_train, y_train = make_moons(n_samples=500, noise=0.1,
                        random_state=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train, y_train)
y_train[y_train == 0] = -1

y_train = y_train

# Set model parameters and initial values
alpha = 1.0
m = len(X_train_scaled)
initial_mu = np.zeros(m)
initial_b = 0.0

# Instantiate model
model_test = SVMC.SMOModel(X = X_train_scaled, y = y_train, alpha = alpha, kernel = SVMC.gaussian_kernel,
                 mu = initial_mu, b = initial_b, errors = np.zeros(m), eps = 1e-2, tol = 1e-2, args = 0.5)

# Initialize error cache
initial_error = SVMC.decision_function(model_test, X_train_scaled) - model_test.y

# initial_error = decision_function2(model_test.mu, model_test.y, model_test.kernel,
                                  # model_test.X, model_test.X, model_test.b) - model_test.y
model_test.errors = initial_error
output = SVMC.train(model_test)
fig, ax = plt.subplots()
grid, ax = plot_decision_boundary(output, ax, sig = model_test.args)

# Test de la fonction metric modifiée 
accuracy, precision, recall, F1,confusion_matrix = SVM.metric(model_test, X_train_scaled)

#%% Test de l'algo des SMO en ayant fait pas mal de modifications 

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def plot_decision_boundary(model, ax, sig, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
        """Plots the model's decision boundary on the input axes object.
        Range of decision boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grid`, `ax`)."""
        
        # Generate coordinate grid of shape [resolution x resolution]
        # and evaluate the model over the entire space
        xrange = np.linspace(model.X[:,0].min(), model.X[:,0].max(), resolution)
        yrange = np.linspace(model.X[:,1].min(), model.X[:,1].max(), resolution)
        grid = [[SVMC.decision_function2(model.mu, model.y,
                                   model.X, np.array([xr, yr]),
                                   model.b, SVMC.gaussian_kernel, sig) for xr in xrange] for yr in yrange]
        grid = np.array(grid).reshape(len(xrange), len(yrange))
        
        # Plot decision contours using grid and
        # make a scatter plot of training data
        ax.contour(xrange, yrange, grid, levels=levels, linewidths=(1, 1, 1),
                   linestyles=('--', '-', '--'), colors=colors)
        ax.scatter(model.X[:,0], model.X[:,1],
                   c=model.y, cmap=plt.cm.viridis, lw=0, alpha=0.25)
        
        # Plot support vectors (non-zero alphas)
        # as circled points (linewidth > 0)
        mask = (np.round(model.mu, decimals=2) != 0.0).flatten()
        ax.scatter(model.X[mask,0], model.X[mask,1],
                   c=model.y[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')
        
        return grid, ax

X_train, y_train = make_moons(n_samples=500, noise=0.1,
                        random_state=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train, y_train)
y_train[y_train == 0] = -1

y_train = y_train.reshape(-1,1)

# Set model parameters and initial values
alpha = 1.0
m = len(X_train_scaled)
initial_mu = np.zeros((m,1))
initial_b = 0.0

# Instantiate model
model_test = SVMC.SMOModel(X_train_scaled, y_train, alpha, SVMC.gaussian_kernel,
                 initial_mu, initial_b, np.zeros(m), 1e-3, 1e-2)


sig = 0.5
for j in range(len(X_train_scaled)):
    # initial_error = SVMC.decision_function2(model.mu, model.y, model.X, model.X[j,:], model.b, SVM.kern_poly, (1.5,3)) - model.y[j][0]
    initial_error = SVMC.decision_function2(model_test.mu, model_test.y, model_test.X, model_test.X[j,:],
                                            model_test.b, SVMC.gaussian_kernel, sig) - model_test.y[j][0]
    model_test.errors[j] = initial_error
# Initialize error cache
# initial_error = decision_function2(model_test.mu, model_test.y, model_test.kernel,
                                  # model_test.X, model_test.X, model_test.b) - model_test.y


K_test = SVM.Gramm(SVM.gauss, X_train_scaled, sig)
# model.errors = initial_error
output = SVMC.train(model_test, K_test, sig)
fig, ax = plt.subplots()
grid, ax = plot_decision_boundary(output, ax, sig = sig)

#%% Endroit momentanément poubelle 

# A = np.argwhere(model.mu[:,0] > 1e-15)
# mu_support = np.zeros(np.shape(A))
# Y_support = np.zeros(np.shape(A))
# X_support = np.zeros((np.shape(A)[0],np.shape(X)[1]))
# for i in range(np.shape(A)[0]):
#     mu_support[i] = output.mu[A[i,0]]
#     Y_support[i] = Y[A[i,0]]
#     X_support[i,:] = X[A[i,0],:]  

# Dy_support = np.diag(Y_support.reshape(-1,))
# G_support = SVM.Gramm(SVM.gauss,X_support)
# bon = 0 
# est = []
# for i in range(len(X_new)):
#     x = X[i,:] 
#     esti = SVM.f_D_data(x, mu_support, Dy_new, b, G_new[i,:].reshape(-1,1))
#     est.append(esti)
#     if np.sign(esti)*Y[i] > 0:
#         bon += 1 
        
# ratio = (bon/np.shape(A)[0])*100