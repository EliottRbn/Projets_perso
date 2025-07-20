# -*- coding: utf-8 -*-
"""

SVM Class for SMO algorithm 

"""

import numpy as np
import numba 

#%% class SMOModel:
"""Container object for the model used for sequential minimal optimization."""
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
    K : Array
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
        print("eta:", eta, "E1:", E1, "E2:", E2, "mu2:", mu2, "y2:", y2)
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
    non_opt = [n for n in range(model.n) if (n != i1 and n != i2)]
    model.errors[non_opt] += y1*(a1-mu1)*update(model.G, i1, non_opt) + \
                             y2*(a2-mu2)*update(model.G, i2, non_opt) + model.b - b_new
    # model.errors[non_opt] = model.errors[non_opt] + \
                            # y1*(a1 - mu1)*model.kernel(model.X[i1], model.X[non_opt],arg) + \
                            # y2*(a2 - mu2)*model.kernel(model.X[i2], model.X[non_opt],arg) + model.b - b_new
    
    # Update model threshold
    model.b = b_new
    
    return 1, model
    
def examine_example(i2, model):
    
    y2 = model.y[i2]
    mu2 = model.mu[i2]
    E2 = model.errors[i2]
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

def train(model):
    """
    Parameters
    ----------
    model : Class object 
        Modèle que l'on souhaite entraîner
    K : Array
        Matrice de Gramm pour le noyau considéré

    Returns
    -------
    model : Class object
        Modèle entraîné
    """
    
    numChanged = 0
    examineAll = 1

    while(numChanged > 0) or (examineAll):
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
def poly_kernel(x,y,args):
    c,d = args 
    result = (x @ y.T + c)**d
    return result

@numba.njit 
def update(G,i,non_opt):
    actualisation = [G[i,k] for k in non_opt]
    return np.array(actualisation)
