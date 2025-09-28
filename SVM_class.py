# -*- coding: utf-8 -*-
"""

SVM Class for SMO algorithm 

"""

import numpy as np
import numba 

#%% 

class SMOModel:
    def __init__(self, X, y, alpha, kernel, mu, b, errors, eps, tol, args):
        self.X = X               # Training data vector
        self.y = y               # Class label vector
        self.alpha = alpha       # Regularization parameter
        self.kernel = kernel     # Kernel function
        self.mu = mu             # Lagrange multiplier vector
        self.b = b               # Scalar bias term
        self.errors = errors     # Error cache
        self.eps = eps           # Mu's tolerance
        self.tol = tol           # Error tolerance
        self._obj = []           # Record of objective function value
        self.n = len(self.X)     # Store size of training set
        self.args = args         # Arguments of the kernel (tuple-like)
        self.G = gramm(self)
    

def gramm(model):
    # G = np.zeros((model.n,model.n))
    # for i in range(model.n):
    #     for j in range(i, model.n):
    #         value = model.kernel(model.X[i,:],model.X[j,:],model.args)
    #         G[i,j] = value 
    #         G[j,i] = value 
    # model.G = G 
    G = model.kernel(model.X,model.X,model.args)
    return G
    
# Objective function to optimize

@numba.njit
def objective_function(mu,y,G):
    """
    Parameters
    ----------
    mu : Array
        Multiplicateurs de Lagrange de notre modèle
    y : Array
        Vecteur qui contient les vraies classes des données
    G : Array
        Matrice de Gramm pour le noyau considéré

    Returns
    -------
    Float
        Fonction coût que l'on cherche à minimiser 

    """
    return np.sum(mu) - 0.5*(mu*y).T@ G @(mu*y)


# Decision function

def decision_function(model,x_test):
    """
    Parameters
    ----------
    model : Class
           Model used to train the SVM classifier 
    x_test : Array
          Image or input we want to evaluate 

    Returns
    -------
    Float
        Prediction of the SVM for the considered input x_test

    """
    
    result = (model.mu * model.y) @ model.kernel(model.X, x_test, model.args) - model.b
    return result

def decision_function2(mu, y, X_train, x_test, b, Kernel, arg):
    """Applies the SVM decision function to the input feature vectors in `x_test`."""
    K = Kernel(X_train,x_test,arg)
    if type(K) == np.ndarray:
        result = (mu * y).T @ (Kernel(X_train, x_test,arg)) - b
    else:
        result = (mu * y).T @ (Kernel(X_train, x_test,arg)) - b
    return result

def take_step(i1, i2, model):
    
    # Skip if chosen mus are the same
    if i1 == i2:
        return 0, model
    
    mu1 = model.mu[i1]
    mu2 = model.mu[i2]
    y1 = model.y[i1]
    y2 = model.y[i2]
    E1 = model.errors[i1]
    E2 = model.errors[i2]
    s = y1 * y2
    
    # Compute L & H, the bounds on new possible mu values
    if (y1 != y2):
        L = max(0, mu2 - mu1)
        H = min(model.alpha, model.alpha + mu2 - mu1)
    elif (y1 == y2):
        L = max(0, mu1 + mu2 - model.alpha)
        H = min(model.alpha, mu1 + mu2)
    if (L == H):
        return 0, model

    # Compute kernel & 2nd derivative eta
    k11 = model.G[i1,i1] 
    k12 = model.G[i1,i2]
    k22 = model.G[i2,i2]
    eta = 2 * k12 - k11 - k22
    
    # Compute new mu 2 (a2) if eta is negative
    if (eta < 0):
        # print("eta:", eta, "E1:", E1, "E2:", E2, "mu2:", mu2, "y2:", y2)
        a2 = mu2 - y2 * (E1 - E2) / eta
        # Clip a2 based on bounds L & H
        if L < a2 < H:
            a2 = a2
        elif (a2 <= L):
            a2 = L
        elif (a2 >= H):
            a2 = H
            
    # If eta is non-negative, move new a2 to bound with greater objective function value
    else:
        mu_adj = model.mu.copy()
        mu_adj[i2] = L
        # objective function output with a2 = L
        Lobj = objective_function(model.mu,model.y,model.G) 
        mu_adj[i2] = H
        # objective function output with a2 = H
        Hobj = objective_function(model.mu,model.y,model.G)
        if Lobj > (Hobj + model.eps):
            a2 = L
        elif Lobj < (Hobj - model.eps):
            a2 = H
        else:
            a2 = mu2
            
    # Push a2 to 0 or C if very close
    if a2 < 1e-8:
        a2 = 0.0
    elif a2 > (model.alpha - 1e-8):
        a2 = model.alpha
    
    # If examples can't be optimized within epsilon (eps), skip this pair
    if (np.abs(a2 - mu2) < model.eps * (a2 + mu2 + model.eps)):
        return 0, model
    
    # Calculate new mu 1 (a1)
    a1 = mu1 + s * (mu2 - a2)
    
    # Update threshold b to reflect newly calculated mus
    # Calculate both possible thresholds
    b1 = E1 + y1 * (a1 - mu1) * k11 + y2 * (a2 - mu2) * k12 + model.b
    b2 = E2 + y1 * (a1 - mu1) * k12 + y2 * (a2 - mu2) * k22 + model.b
    
    # Set new threshold based on if a1 or a2 is bound by L and/or H
    if 0 < a1 and a1 < model.alpha:
        b_new = b1
    elif 0 < a2 and a2 < model.alpha:
        b_new = b2
    # Average thresholds if both are bound
    else:
        b_new = (b1 + b2) * 0.5

    # Update model object with new mus & threshold
    model.mu[i1] = a1
    model.mu[i2] = a2
    
    # Update error cache
    # Error cache for optimized mus is set to 0 if they're unbound
    for index, mu_ in zip([i1, i2], [a1, a2]):
        if 0.0 < mu_ < model.alpha:
            model.errors[index] = 0.0
    
    # Set non-optimized errors based on equation 12.11 in Platt's book
    non_opt = np.array([n for n in range(model.n) if (n != i1 and n != i2)], dtype = np.int64)
    model.errors[non_opt] += y1*(a1-mu1)*update(model.G, i1, non_opt) + \
                             y2*(a2-mu2)*update(model.G, i2, non_opt) + model.b - b_new
    
    # Update model threshold
    model.b = b_new
    
    return 1, model
    
def examine_example(i2, model):
    
    y2 = model.y[i2]
    mu2 = model.mu[i2]
    E2 = model.errors[i2]
    
    if np.abs(E2) > 1e9:  
        raise ValueError
    
    r2 = E2 * y2

    # Proceed if error is within specified tolerance (tol)
    if ((r2 < -model.tol and mu2 < model.alpha) or (r2 > model.tol and mu2 > 0)):
        
        if len(model.mu[(model.mu != 0) & (model.mu != model.alpha)]) > 1:
            # Use 2nd choice heuristic is choose max difference in error
            if model.errors[i2] > 0:
                i1 = np.argmin(model.errors)
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
            
        # Loop through non-zero and non-alpha mus, starting at a random point
        for i1 in np.roll(np.where((model.mu != 0) & (model.mu != model.alpha))[0],
                          np.random.choice(np.arange(model.n))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
        
        # loop through all mus, starting at a random point
        for i1 in np.roll(np.arange(model.n), np.random.choice(np.arange(model.n))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
    
    return 0, model

def train(model, max_iter = 100000):
    """
    Parameters
    ----------
    model : Class object 
        Modèle que l'on souhaite entraîner
    max_iter : Int
        Nombre maximum d'itérations de l'algorithme pour le stopper s'il ne parvient pas à séparer les données

    Returns
    -------
    model : Class object
        Modèle entraîné
    """
    
    numChanged = 0
    examineAll = 1
    iteration = 0
    
    try:
        while(numChanged > 0) or (examineAll) and (iteration < max_iter):
            numChanged = 0
            if examineAll:
                # loop over all training examples
                for i in range(model.mu.shape[0]):
                    examine_result, model = examine_example(i, model)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = objective_function(model.mu,model.y,model.G)
                        model._obj.append(obj_result)
                
            else:
                # loop over examples where mus are not already at their limits
                for i in np.where((model.mu != 0) & (model.mu != model.alpha))[0]:
                    examine_result, model = examine_example(i, model)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = objective_function(model.mu,model.y,model.G)
                        model._obj.append(obj_result)
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1
            
            iteration += 1
        
        if iteration == max_iter:
            print("Le nombre maximum d'itération a été atteint")
            
    except ValueError:
        print("L'algorithme diverge, arrêt automatique de l'entraînement\n")
    
    return model

# Code fait avec des boucles (donc plus long et moins propre) pour l'optimisation numba.njit (sans les np.newaxis, np.linalg.norm...)
@numba.njit(parallel = True)
def gaussian_kernel(x, y, args):
    """Returns the gaussian similarity of arrays `x` and `y` with
    kernel width parameter `sigma` (set to 10 by default)."""
    if x.ndim == 1 and y.ndim == 1:
        result = np.exp(-((x-y)@(x-y))/ (2 * args ** 2))
        
    elif (x.ndim > 1 and y.ndim == 1) or (x.ndim == 1 and y.ndim > 1):
        diff = x - y
        dist2 = np.sum(diff**2, axis=1)
        result = np.exp(-dist2 / (2 * args **2))
        
    elif x.ndim > 1 and y.ndim > 1:
        n, m = x.shape[0], y.shape[0]
        result = np.empty((n, m))
        for i in numba.prange(n): # Parallélisation de la boucle sur les différents coeurs du CPU pour accélérer encore plus le process
            for j in range(m):
                diff = x[i,:] - y[j,:]  
                result[i, j] = np.exp(-np.dot(diff, diff) / (2 * args ** 2))
                
    return result

@numba.njit
def linear_kernel(x, y, args = 1):
    """Returns the linear combination of arrays `x` and `y` with
    the optional bias term `b` ( = args ) (set to 1 by default)."""
    
    return x @ y.T + args # Note the @ operator for matrix multiplication

@numba.njit
def poly_kernel(x, y, args):
    c,d = args 
    result = (x @ y.T + c)**d
    return result

@numba.njit 
def update(G, i, non_opt):
    n = len(non_opt)
    actualisation = np.empty(n, dtype=G.dtype)  # allocation directe
    for idx in range(n):
        actualisation[idx] = G[i, non_opt[idx]]
    return actualisation

#%% 


"""

Test d'optimisation de l'algorithme de SMO via une étape de filtrage des données en se basant sur une thèse. 
Le test n'est pas encore concluant, l'algorithme est bien trop lent et ne semble pas filtrer les données comme voulu.

"""

#%% 

"""
def covered_hyperplan(x_i, x_k, x, kernel, args):
    D = kernel(x_i, x_i, args) + kernel(x_k, x_k, args) - 2*kernel(x_i, x_k, args)
    if D == 0:
        return 0 # Éviter de diviser par 0
    b = (kernel(x_i, x_i, args) - kernel(x_k, x_k, args))/D
    return (1/D)*(kernel(x_i, x, args) - kernel(x_k, x, args)) + b

def covered_examples(X, Y, kernel, args, rho=0.1, eps=1e-12):
    
    CEFA vectorisé et robuste.
    - X: (n,d), Y: (n,), labels dans {+1, -1}
    - kernel: fonction vectorisée (supporte X vs X)
    - rho: degré de filtrage
    
    n = X.shape[0]
    K = kernel(X, X, args)
    diagK = np.diag(K)

    pos_idx = np.where(Y > 0)[0]
    neg_idx = np.where(Y < 0)[0]

    kept_pos, removed_pos = [], []
    kept_neg, removed_neg = [], []

    # ----------- filtrage classe positive -----------
    for i in pos_idx:
        J = np.delete(pos_idx, np.where(pos_idx == i)[0])
        if J.size == 0:
            kept_pos.append(i)
            continue

        remove_i = False
        for k in neg_idx:
            D_ik = K[i, i] + K[k, k] - 2 * K[i, k]
            if abs(D_ik) < eps:
                continue

            H_ik = (K[i, J] - K[k, J]) / D_ik + (K[i, i] - K[k, k]) / D_ik

            D_Jk = diagK[J] + K[k, k] - 2 * K[J, k]
            safe = np.abs(D_Jk) >= eps
            if not np.any(safe):
                continue

            H_jk = np.full_like(D_Jk, np.inf, dtype=float)
            H_jk[safe] = (K[J[safe], i] - K[k, i]) / D_Jk[safe] + (diagK[J[safe]] - K[k, k]) / D_Jk[safe]

            mask_ik = (rho - 1 <= H_ik) & (H_ik <= 1 - rho)
            if not np.all(mask_ik):  # optimisation
                continue
            mask_jk = (H_jk < rho - 1) | (H_jk > 1 - rho)

            if np.all(mask_ik & mask_jk):
                remove_i = True
                break

        (removed_pos if remove_i else kept_pos).append(i)

    kept_pos = np.array(kept_pos, dtype=int)

    # ----------- filtrage classe négative -----------
    for i in neg_idx:
        J = np.delete(neg_idx, np.where(neg_idx == i)[0])
        if J.size == 0:
            kept_neg.append(i)
            continue

        remove_i = False
        for k in kept_pos:  # seulement les positifs gardés
            D_ik = K[i, i] + K[k, k] - 2 * K[i, k]
            if abs(D_ik) < eps:
                continue

            H_ik = (K[i, J] - K[k, J]) / D_ik + (K[i, i] - K[k, k]) / D_ik

            D_Jk = diagK[J] + K[k, k] - 2 * K[J, k]
            safe = np.abs(D_Jk) >= eps
            if not np.any(safe):
                continue

            H_jk = np.full_like(D_Jk, np.inf, dtype=float)
            H_jk[safe] = (K[J[safe], i] - K[k, i]) / D_Jk[safe] + (diagK[J[safe]] - K[k, k]) / D_Jk[safe]

            mask_ik = (rho - 1 <= H_ik) & (H_ik <= 1 - rho)
            if not np.all(mask_ik):
                continue
            mask_jk = (H_jk < rho - 1) | (H_jk > 1 - rho)

            if np.all(mask_ik & mask_jk):
                remove_i = True
                break

        (removed_neg if remove_i else kept_neg).append(i)

    # ----------- assemblage des résultats -----------
    kept_idx = np.concatenate([np.array(kept_pos, dtype=int),
                               np.array(kept_neg, dtype=int)]) if (len(kept_pos) or len(kept_neg)) else np.array([], dtype=int)

    removed_idx = np.concatenate([np.array(removed_pos, dtype=int),
                                  np.array(removed_neg, dtype=int)]) if (len(removed_pos) or len(removed_neg)) else np.array([], dtype=int)

    X_filtered = X[kept_idx] if kept_idx.size else np.empty((0, X.shape[1]))
    Y_filtered = Y[kept_idx] if kept_idx.size else np.empty((0,), dtype=Y.dtype)

    X_removed = X[removed_idx] if removed_idx.size else np.empty((0, X.shape[1]))
    Y_removed = Y[removed_idx] if removed_idx.size else np.empty((0,), dtype=Y.dtype)

    return X_filtered, Y_filtered, X_removed, Y_removed


def covered_examples(X, Y, kernel, args, rho):
    # rho est le degré de filtrage 
    X_pos = X[Y > 0]
    X_neg = X[Y < 0]
    
    X_pos_filtred = []
    X_pos_removed = []
    
    X_neg_filtred = []
    X_neg_removed = []
    
    i = 0
    while i < len(X_pos):
        print(len(X_pos_filtred))
        x_i = X_pos[i]
        print(f"Examen de l'exemple positif {x_i}")
        X_pos_wo_xi = np.delete(X_pos, i, axis=0)  # retirer x_i
    
        remove_i = False  # flag pour savoir si on va supprimer x_i
        non_ecart = 0
        # for x_k in X_neg_filtred:
        #     H_ik = covered_hyperplan(x_i, x_k, X_pos_wo_xi, kernel, args)
        #     H_jk = covered_hyperplan(X_pos_wo_xi, x_k, x_i, kernel, args)
            
        #     # for j, x_j in enumerate(X_pos_wo_xi):
        #     #     H_jk = covered_hyperplan(x_j, x_k, x_i, kernel, args)
        #     if (rho - 1 <= H_ik.any() <= 1 - rho) and ((rho - 1 > H_jk.any()) or (H_jk.any() > 1 - rho)):
        #         non_ecart += 1
        for x_k in X_neg:
            H_ik = covered_hyperplan(x_i, x_k, X_pos_wo_xi, kernel, args)  # -> vecteur
            H_jk = covered_hyperplan(X_pos_wo_xi, x_k, x_i, kernel, args)  # -> vecteur
        
            mask_ik = (rho - 1 <= H_ik) & (H_ik <= 1 - rho)
            mask_jk = (H_jk < rho - 1) | (H_jk > 1 - rho)
            if np.any(mask_ik & mask_jk):
                non_ecart += 1
            
            if non_ecart == X_pos_wo_xi.shape[0]:
                remove_i = True
                break
        
        if remove_i:
            # ne pas incrémenter i car on a réduit le tableau
            X_pos_removed.append(x_i.copy())
            print("Une donnée de la classe positive a été filtrée")
            i += 1
        else:
            i += 1  # continuer avec l'élément suivant
    
    for x_pos in X_pos:
        if x_pos not in X_pos_removed:
            X_pos_filtred.append(x_pos)
    
    i = 0
    while i < len(X_neg):
        print(len(X_neg_filtred))
        x_i = X_neg[i]
        print(f"Examen de l'exemple négatif {x_i}")
        X_neg_wo_xi = np.delete(X_neg, i, axis=0)  # retirer x_i
        
        remove_i = False  # flag pour savoir si on va supprimer x_i
        for x_k in X_pos_filtred:
            non_ecart = 0
            H_ik = covered_hyperplan(x_i, x_k, X_neg_wo_xi, kernel, args)
            
            for j, x_j in enumerate(X_neg_wo_xi):
                H_jk = covered_hyperplan(x_j, x_k, x_i, kernel, args)
                if (rho - 1 <= H_ik[j] <= 1 - rho) and ((rho - 1 > H_jk) or (H_jk > 1 - rho)):
                    non_ecart += 1
                    break
            
            if non_ecart == X_neg_wo_xi.shape[0]:
                remove_i = True
                break
        
        if remove_i:
            # ne pas incrémenter i car on a réduit le tableau
            X_neg_removed.append(x_i.copy())
            i += 1
            print("Une donnée de la classe négative a été filtrée")
        else:
            i += 1  # continuer avec l'élément suivant
    
    for x_neg in X_neg:
        if x_neg not in X_neg_removed:
            X_neg_filtred.append(x_neg)
    
    Y_pos_filtred = np.ones(len(X_pos_filtred))
    Y_pos_removed = np.ones(len(X_pos_removed))
    
    Y_neg_filtred = np.ones(len(X_neg_filtred))
    Y_neg_removed = np.ones(len(X_neg_removed))
    
    X_filtred = np.concatenate((np.array(X_pos_filtred), np.array(X_neg_filtred)))
    Y_filtred = np.concatenate((Y_pos_filtred, Y_neg_filtred))
    X_non_used = np.concatenate((np.array(X_pos_removed), np.array(X_neg_removed)))
    Y_non_used = np.concatenate((Y_pos_removed, Y_neg_removed))
    
    return X_filtred,Y_filtred, X_non_used, Y_non_used 
"""
