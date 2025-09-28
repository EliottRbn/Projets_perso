# -*- coding: utf-8 -*-
"""

Test sur les différents descripteurs et différents noyaux 

"""

# Import des libraires 

import numpy as np
import SVM_lib as SVM
import SVM_class as SVMC 
import importlib

#%% 

### Chargement des données ###

X_raw,Y_raw = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 500, 500, config = SVM.raw_data)
X_HOG,Y_HOG = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 500, 500, config = SVM.HOG_data)
X_Haar,Y_Haar = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 500, 500, config = SVM.Haar_data)
X_LBP,Y_LBP = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 500, 500, config = SVM.LBP_data)
X_Haar_HOG,Y_Haar_HOG = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 500, 500, config = SVM.HOG_Haar_data)
X_HOG_LBP,Y_HOG_LBP = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 500, 500, config = SVM.HOG_LBP_data)
X_Haar_LBP,Y_Haar_LBP = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 500, 500, config = SVM.Haar_LBP_data)
X_Haar_HOG_LBP,Y_Haar_HOG_LBP = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 500, 500, config = SVM.Haar_HOG_LBP_data)

### Paramètres des SMO ### 

alpha = 0.8
mu = np.zeros(X_raw.shape[0])
b = 0
errors = np.zeros(X_raw.shape[0]) 
eps = 1e-2 
tol = 1e-2
args = (0,1)

### Sans noyau gaussien pour débuter i.e. SVM Linéaires ###

## Initialisation des SVM ##

model_raw = SVMC.SMOModel(X_raw, Y_raw, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
model_HOG = SVMC.SMOModel(X_HOG, Y_HOG, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
model_Haar = SVMC.SMOModel(X_Haar, Y_Haar, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
model_LBP = SVMC.SMOModel(X_LBP, Y_LBP, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
model_Haar_HOG = SVMC.SMOModel(X_Haar_HOG, Y_Haar_HOG, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
model_HOG_LBP = SVMC.SMOModel(X_HOG_LBP, Y_HOG_LBP, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
model_Haar_LBP = SVMC.SMOModel(X_Haar_LBP, Y_Haar_LBP, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
model_Haar_HOG_LBP = SVMC.SMOModel(X_Haar_HOG_LBP, Y_Haar_HOG_LBP, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)

## Initialisation des erreurs ## 

initial_error = SVMC.decision_function(model_raw, X_raw) - model_raw.y
model_raw.errors = initial_error

initial_error = SVMC.decision_function(model_HOG, X_HOG) - model_HOG.y
model_HOG.errors = initial_error

initial_error = SVMC.decision_function(model_Haar, X_Haar) - model_Haar.y
model_Haar.errors = initial_error

initial_error = SVMC.decision_function(model_LBP, X_LBP) - model_LBP.y
model_LBP.errors = initial_error

initial_error = SVMC.decision_function(model_Haar_HOG, X_Haar_HOG) - model_Haar_HOG.y
model_Haar_HOG.errors = initial_error

initial_error = SVMC.decision_function(model_HOG_LBP, X_HOG_LBP) - model_HOG_LBP.y
model_HOG_LBP.errors = initial_error

initial_error = SVMC.decision_function(model_Haar_LBP, X_Haar_LBP) - model_Haar_LBP.y
model_Haar_LBP.errors = initial_error

initial_error = SVMC.decision_function(model_Haar_HOG_LBP, X_Haar_HOG_LBP) - model_Haar_HOG_LBP.y
model_Haar_HOG_LBP.errors = initial_error

## Training des modèles ## 

print("Entraînement du modèle avec les data en brut")
model_raw = SVMC.train(model_raw)

print("Entraînement du modèle avec le descripteur HOG")
model_HOG = SVMC.train(model_HOG)

print("Entraînement du modèle avec le descripteur Haar")
model_Haar = SVMC.train(model_Haar)

print("Entraînement du modèle avec le descripteur LBP")
model_LBP = SVMC.train(model_LBP)

print("Entraînement du modèle avec le mixte de Haar et HOG")
model_Haar_HOG = SVMC.train(model_Haar_HOG)

print("Entraînement du modèle avec le mixte de HOG et LBP")
model_HOG_LBP = SVMC.train(model_HOG_LBP)

print("Entraînement du modèle avec le mixte de Haar et LBP")
model_Haar_LBP = SVMC.train(model_Haar_LBP)

print("Entraînement du modèle avec le mixte de Haar, HOG et LBP")
model_Haar_HOG_LBP = SVMC.train(model_Haar_HOG_LBP)

# import importlib 
# importlib.reload(SVM)

### Évaluation des performances des modèles ###

models = [[model_raw,model_raw.X,model_raw.y,"les datas bruts", SVM.raw_data],
          [model_HOG,model_HOG.X,model_HOG.y,"HOG", SVM.HOG_data],
          [model_Haar,model_Haar.X,model_Haar.y,"Haar", SVM.Haar_data],
          [model_LBP, model_LBP.X, model_LBP.y, "LBP", SVM.LBP_data],
          [model_Haar_HOG,model_Haar_HOG.X,model_Haar_HOG.y,"le mixte de Haar et HOG", SVM.HOG_Haar_data],
          [model_HOG_LBP,model_HOG_LBP.X,model_HOG_LBP.y,"le mixte de HOG et LBP", SVM.HOG_LBP_data],
          [model_Haar_LBP,model_Haar_LBP.X,model_Haar_LBP.y,"le mixte de Haar et LBP", SVM.Haar_LBP_data],
          [model_Haar_HOG_LBP,model_Haar_HOG_LBP.X,model_Haar_HOG_LBP.y,"le mixte de Haar, HOG et LBP", SVM.Haar_HOG_LBP_data]]

for model in models:
    accuracy, precision, recall, F1,confusion_matrix = SVM.metric(model[0], model[1], model[2])
    print(f"Résultats pour le model avec {model[3]} : \n")
    print("Accuracy = ", np.round(accuracy*100,2))
    print("Precision = ", np.round(precision*100,2))
    print("Recall = ", np.round(recall*100,2))
    print("F1 = ", np.round(F1*100,2))
    
#%% Métriques sur des BDD test 

X_raw_test,Y_raw_test = SVM.routine_test('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 1000, 1000, config = SVM.raw_data)
X_raw_test = np.concatenate((X_raw_test[500:1000,:],X_raw_test[1500:,:]))
Y_raw_test = np.concatenate((Y_raw_test[500:1000], Y_raw_test[1500:]))

X_Haar_test,Y_Haar_test = SVM.routine_test('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 1000, 1000, config = SVM.Haar_data)
X_Haar_test = np.concatenate((X_Haar_test[500:1000,:],X_Haar_test[1500:,:]))
Y_Haar_test = np.concatenate((Y_Haar_test[500:1000], Y_Haar_test[1500:]))

X_HOG_test,Y_HOG_test = SVM.routine_test('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 1000, 1000, config = SVM.HOG_data)
X_HOG_test = np.concatenate((X_HOG_test[500:1000,:],X_HOG_test[1500:,:]))
Y_HOG_test = np.concatenate((Y_HOG_test[500:1000], Y_HOG_test[1500:]))

X_LBP_test,Y_LBP_test = SVM.routine_test('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 1000, 1000, config = SVM.LBP_data)
X_LBP_test = np.concatenate((X_LBP_test[500:1000,:],X_LBP_test[1500:,:]))
Y_LBP_test = np.concatenate((Y_LBP_test[500:1000], Y_LBP_test[1500:]))

X_Haar_HOG_test,Y_Haar_HOG_test = SVM.routine_test('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 1000, 1000, config = SVM.HOG_Haar_data)
X_Haar_HOG_test = np.concatenate((X_Haar_HOG_test[500:1000,:],X_Haar_HOG_test[1500:,:]))
Y_Haar_HOG_test = np.concatenate((Y_Haar_HOG_test[500:1000], Y_Haar_HOG_test[1500:]))

X_Haar_LBP_test,Y_Haar_LBP_test = SVM.routine_test('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 1000, 1000, config = SVM.Haar_LBP_data)
X_Haar_LBP_test = np.concatenate((X_Haar_LBP_test[500:1000,:],X_Haar_LBP_test[1500:,:]))
Y_Haar_LBP_test = np.concatenate((Y_Haar_LBP_test[500:1000], Y_Haar_LBP_test[1500:]))

X_HOG_LBP_test,Y_HOG_LBP_test = SVM.routine_test('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 1000, 1000, config = SVM.HOG_LBP_data)
X_HOG_LBP_test = np.concatenate((X_HOG_LBP_test[500:1000,:],X_HOG_LBP_test[1500:,:]))
Y_HOG_LBP_test = np.concatenate((Y_HOG_LBP_test[500:1000], Y_HOG_LBP_test[1500:]))

X_Haar_HOG_LBP_test,Y_Haar_HOG_LBP_test = SVM.routine_test('BD/train', 'BD/cifar-10-batches-py/data_batch_1', 1000, 1000, config = SVM.Haar_HOG_LBP_data)
X_Haar_HOG_LBP_test = np.concatenate((X_Haar_HOG_LBP_test[500:1000,:],X_Haar_HOG_LBP_test[1500:,:]))
Y_Haar_HOG_LBP_test = np.concatenate((Y_Haar_HOG_LBP_test[500:1000], Y_Haar_HOG_LBP_test[1500:]))

models = [[model_raw,X_raw_test,Y_raw_test,"les datas bruts", SVM.raw_data],
          [model_HOG,X_HOG_test,Y_HOG_test,"HOG", SVM.HOG_data],
          [model_Haar,X_Haar_test,Y_Haar_test,"Haar", SVM.Haar_data],
          [model_LBP, X_LBP_test, Y_LBP_test, "LBP", SVM.LBP_data],
          [model_Haar_HOG,X_Haar_HOG_test, Y_Haar_HOG_test,"le mixte de Haar et HOG", SVM.HOG_Haar_data],
          [model_HOG_LBP,X_HOG_LBP_test,Y_HOG_LBP_test,"le mixte de HOG et LBP", SVM.HOG_LBP_data],
          [model_Haar_LBP,X_Haar_LBP_test,Y_Haar_LBP_test,"le mixte de Haar et LBP", SVM.Haar_LBP_data],
          [model_Haar_HOG_LBP,X_Haar_HOG_LBP_test,Y_Haar_HOG_LBP_test,"le mixte de Haar, HOG et LBP", SVM.Haar_HOG_LBP_data]]

for model in models:
    accuracy, precision, recall, F1,confusion_matrix = SVM.metric(model[0], model[1], model[2])
    print(f"Résultats pour le model sur un BDD test avec {model[3]} : \n")
    print("Accuracy = ", np.round(accuracy*100,2))
    print("Precision = ", np.round(precision*100,2))
    print("Recall = ", np.round(recall*100,2))
    print("F1 = ", np.round(F1*100,2))

#%% Pour un nombre total de 10000 images 

n = 10000 # Pas tous en même temps au risque d'absolument tout faire planter

# X_raw_test,Y_raw_test = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', n, n, config = SVM.raw_data)
# X_raw = np.concatenate((X_raw_test[:n//2,:],X_raw_test[n:(n//2 + n),:]))
# Y_raw = np.concatenate((Y_raw_test[:n//2],Y_raw_test[n:(n//2 + n)]))
# X_raw_test = np.concatenate((X_raw_test[n//2:n,:],X_raw_test[(n//2 + n):,:]))
# Y_raw_test = np.concatenate((Y_raw_test[n//2:n], Y_raw_test[(n//2 + n):]))

# X_Haar_test,Y_Haar_test = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', n, n, config = SVM.Haar_data)
# X_Haar = np.concatenate((X_Haar_test[:n//2,:],X_Haar_test[n:(n//2 + n),:]))
# Y_Haar = np.concatenate((Y_Haar_test[:n//2],Y_Haar_test[n:(n//2 + n)]))
# X_Haar_test = np.concatenate((X_Haar_test[n//2:n,:],X_Haar_test[(n//2 + n):,:]))
# Y_Haar_test = np.concatenate((Y_Haar_test[n//2:n], Y_Haar_test[(n//2 + n):]))

# X_HOG_test,Y_HOG_test = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', n, n, config = SVM.HOG_data)
# X_HOG = np.concatenate((X_HOG_test[:n//2,:],X_HOG_test[n:(n//2 + n),:]))
# Y_HOG = np.concatenate((Y_HOG_test[:n//2],Y_HOG_test[n:(n//2 + n)]))
# X_HOG_test = np.concatenate((X_HOG_test[n//2:n,:],X_HOG_test[(n//2 + n):,:]))
# Y_HOG_test = np.concatenate((Y_HOG_test[n//2:n], Y_HOG_test[(n//2 + n):]))

# X_LBP_test,Y_LBP_test = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', n, n, config = SVM.LBP_data)
# X_LBP = np.concatenate((X_LBP_test[:n//2,:],X_LBP_test[n:(n//2 + n),:]))
# Y_LBP = np.concatenate((Y_LBP_test[:n//2],Y_LBP_test[n:(n//2 + n)]))
# X_LBP_test = np.concatenate((X_LBP_test[n//2:n,:],X_LBP_test[(n//2 + n):,:]))
# Y_LBP_test = np.concatenate((Y_LBP_test[n//2:n], Y_LBP_test[(n//2 + n):]))

# X_Haar_HOG_test,Y_Haar_HOG_test = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', n, n, config = SVM.HOG_Haar_data)
# X_Haar_HOG = np.concatenate((X_Haar_HOG_test[:n//2,:],X_Haar_HOG_test[n:(n//2 + n),:]))
# Y_Haar_HOG = np.concatenate((Y_Haar_HOG_test[:n//2],Y_Haar_HOG_test[n:(n//2 + n)]))
# X_Haar_HOG_test = np.concatenate((X_Haar_HOG_test[n//2:n,:],X_Haar_HOG_test[(n//2 + n):,:]))
# Y_Haar_HOG_test = np.concatenate((Y_Haar_HOG_test[n//2:n], Y_Haar_HOG_test[(n//2 + n):]))

X_Haar_LBP_test,Y_Haar_LBP_test = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', n, n, config = SVM.Haar_LBP_data)
X_Haar_LBP = np.concatenate((X_Haar_LBP_test[:n//2,:],X_Haar_LBP_test[n:(n//2 + n),:]))
Y_Haar_LBP = np.concatenate((Y_Haar_LBP_test[:n//2],Y_Haar_LBP_test[n:(n//2 + n)]))
X_Haar_LBP_test = np.concatenate((X_Haar_LBP_test[n//2:n,:],X_Haar_LBP_test[(n//2 + n):,:]))
Y_Haar_LBP_test = np.concatenate((Y_Haar_LBP_test[n//2:n], Y_Haar_LBP_test[(n//2 + n):]))

X_HOG_LBP_test,Y_HOG_LBP_test = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', n, n, config = SVM.HOG_LBP_data)
X_HOG_LBP = np.concatenate((X_HOG_LBP_test[:n//2,:],X_HOG_LBP_test[n:(n//2 + n),:]))
Y_HOG_LBP = np.concatenate((Y_HOG_LBP_test[:n//2],Y_HOG_LBP_test[n:(n//2 + n)]))
X_HOG_LBP_test = np.concatenate((X_HOG_LBP_test[n//2:n,:],X_HOG_LBP_test[(n//2 + n):,:]))
Y_HOG_LBP_test = np.concatenate((Y_HOG_LBP_test[n//2:n], Y_HOG_LBP_test[(n//2 + n):]))

X_Haar_HOG_LBP_test,Y_Haar_HOG_LBP_test = SVM.routine_test_padded('BD/train', 'BD/cifar-10-batches-py/data_batch_1', n, n, config = SVM.Haar_HOG_LBP_data)
X_Haar_HOG_LBP = np.concatenate((X_Haar_HOG_LBP_test[:n//2,:],X_Haar_HOG_LBP_test[n:(n//2 + n),:]))
Y_Haar_HOG_LBP = np.concatenate((Y_Haar_HOG_LBP_test[:n//2],Y_Haar_HOG_LBP_test[n:(n//2 + n)]))
X_Haar_HOG_LBP_test = np.concatenate((X_Haar_HOG_LBP_test[n//2:n,:],X_Haar_HOG_LBP_test[(n//2 + n):,:]))
Y_Haar_HOG_LBP_test = np.concatenate((Y_Haar_HOG_LBP_test[n//2:n], Y_Haar_HOG_LBP_test[(n//2 + n):]))

### Paramètres des SMO ### 

alpha = 0.8
mu = np.zeros(n)
b = 0
errors = np.zeros(n) 
eps = 1e-2 
tol = 1e-2
args = (0,1)

### Sans noyau gaussien pour débuter i.e. SVM Linéaires ###

## Initialisation des SVM ##

# model_raw = SVMC.SMOModel(X_raw, Y_raw, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
# model_HOG = SVMC.SMOModel(X_HOG, Y_HOG, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
# model_Haar = SVMC.SMOModel(X_Haar, Y_Haar, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
# model_LBP = SVMC.SMOModel(X_LBP, Y_LBP, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
# model_Haar_HOG = SVMC.SMOModel(X_Haar_HOG, Y_Haar_HOG, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
model_HOG_LBP = SVMC.SMOModel(X_HOG_LBP, Y_HOG_LBP, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
model_Haar_LBP = SVMC.SMOModel(X_Haar_LBP, Y_Haar_LBP, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)
model_Haar_HOG_LBP = SVMC.SMOModel(X_Haar_HOG_LBP, Y_Haar_HOG_LBP, alpha, SVMC.poly_kernel, mu, b, errors, eps, tol, args)

## Initialisation des erreurs ## 

# initial_error = SVMC.decision_function(model_raw, X_raw) - model_raw.y
# model_raw.errors = initial_error

# initial_error = SVMC.decision_function(model_HOG, X_HOG) - model_HOG.y
# model_HOG.errors = initial_error

# initial_error = SVMC.decision_function(model_Haar, X_Haar) - model_Haar.y
# model_Haar.errors = initial_error

# initial_error = SVMC.decision_function(model_LBP, X_LBP) - model_LBP.y
# model_LBP.errors = initial_error

# initial_error = SVMC.decision_function(model_Haar_HOG, X_Haar_HOG) - model_Haar_HOG.y
# model_Haar_HOG.errors = initial_error

initial_error = SVMC.decision_function(model_HOG_LBP, X_HOG_LBP) - model_HOG_LBP.y
model_HOG_LBP.errors = initial_error

initial_error = SVMC.decision_function(model_Haar_LBP, X_Haar_LBP) - model_Haar_LBP.y
model_Haar_LBP.errors = initial_error

initial_error = SVMC.decision_function(model_Haar_HOG_LBP, X_Haar_HOG_LBP) - model_Haar_HOG_LBP.y
model_Haar_HOG_LBP.errors = initial_error

## Training des models ## 

# print("Entraînement du modèle avec les data en brut")
# model_raw = SVMC.train(model_raw)

# print("Entraînement du modèle avec le descripteur HOG")
# model_HOG = SVMC.train(model_HOG)

# print("Entraînement du modèle avec le descripteur Haar")
# model_Haar = SVMC.train(model_Haar)

# print("Entraînement du modèle avec le descripteur LBP")
# model_LBP = SVMC.train(model_LBP)

# print("Entraînement du modèle avec le mixte de Haar et HOG")
# model_Haar_HOG = SVMC.train(model_Haar_HOG)

print("Entraînement du modèle avec le mixte de HOG et LBP")
model_HOG_LBP = SVMC.train(model_HOG_LBP)

print("Entraînement du modèle avec le mixte de Haar et LBP")
model_Haar_LBP = SVMC.train(model_Haar_LBP)

print("Entraînement du modèle avec le mixte de Haar, HOG et LBP")
model_Haar_HOG_LBP = SVMC.train(model_Haar_HOG_LBP)

models = """[model_raw,model_raw.X,model_raw.y,"les datas bruts", SVM.raw_data],
          [model_HOG,model_HOG.X,model_HOG.y,"HOG", SVM.HOG_data],
          [model_Haar,model_Haar.X,model_Haar.y,"Haar", SVM.Haar_data],
          [model_LBP, model_LBP.X, model_LBP.y, "LBP", SVM.LBP_data],
          [model_Haar_HOG,model_Haar_HOG.X,model_Haar_HOG.y,"le mixte de Haar et HOG", SVM.HOG_Haar_data],"""
models =          [[model_HOG_LBP,model_HOG_LBP.X,model_HOG_LBP.y,"le mixte de HOG et LBP", SVM.HOG_LBP_data],
          [model_Haar_LBP,model_Haar_LBP.X,model_Haar_LBP.y,"le mixte de Haar et LBP", SVM.Haar_LBP_data],
          [model_Haar_HOG_LBP,model_Haar_HOG_LBP.X,model_Haar_HOG_LBP.y,"le mixte de Haar, HOG et LBP", SVM.Haar_HOG_LBP_data]]

for model in models:
    accuracy, precision, recall, F1,confusion_matrix = SVM.metric(model[0], model[1], model[2])
    print(f"Résultats pour le model avec {model[3]} : \n")
    print("Accuracy = ", np.round(accuracy*100,2))
    print("Precision = ", np.round(precision*100,2))
    print("Recall = ", np.round(recall*100,2))
    print("F1 = ", np.round(F1*100,2))