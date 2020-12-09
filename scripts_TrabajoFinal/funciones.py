# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:12:15 2020

@author: Pablo Saldarriaga
"""
import math
import numpy as np
import random
from sklearn.base import clone
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
#%%
def precision_recall_auc_score(y_true, proba):

    precision, recall, _ = metrics.precision_recall_curve(y_true, proba)

    auc = metrics.auc(recall, precision)

    return auc


def sol2array(individuo, bin_cont = False):
    max_lay = 4
    sol = list(np.zeros(4 + max_lay))

    eq = {'activation': {"identity": 1, "logistic": 2, "tanh": 3, "relu": 4}}

    sol[4] = eq['activation'][individuo['activation']]
    
    if bin_cont:
        sol[1] = individuo['alpha_bin']
        sol[2] = individuo['learning_rate_init_bin']
    else:
        sol[1] = individuo['alpha']
        sol[2] = individuo['learning_rate_init']
    
    sol[3] = len(individuo['layer_sizes'])

    for i in range(len(individuo['layer_sizes'])):
        sol[i + 4] = individuo['layer_sizes'][i]

    return sol


def array2sol(sol, bin_cont = False):

    eq = {'activation': {1: "identity", 2: "logistic", 3: "tanh", 4: "relu"}}

    activation = eq['activation'][int(sol[0])]
    alpha = sol[1]
    lr = sol[2]

    layers = []
    for i in range(4, len(sol)):
        if sol[i] != 0:
            layers.append(int(sol[i]))
    
    if bin_cont:
        lr_bin = lr
        alpha_bin = alpha
        
        ### PARAMETROS QUEMADOS EN EL CODIGO, ACA VAN MAX LR Y MAX ALPHA
        lr = bin2cont(lr, 10)
        alpha = bin2cont(alpha, 1)
        
        solucion = {
            "layer_sizes": tuple(layers),
            "activation": activation,
            "alpha": alpha,
            "alpha_bin": alpha_bin,
            "learning_rate_init": lr,
            "learning_rate_init_bin": lr_bin
        }
    else:
        solucion = {
            "layer_sizes": tuple(layers),
            "activation": activation,
            "alpha": alpha,
            "learning_rate_init": lr,
        }

    return solucion

def estandatizar_sol(sol):
    
    new_sol = sol.copy()
    
    new_sol[0] = new_sol[0]/2
    new_sol[1] = new_sol[1]/10
    new_sol[2] = new_sol[2]/1
    new_sol[3] = new_sol[2]/4
    
    for i in range(4,len(new_sol)):
        new_sol[i] = new_sol[i]/128
    
    return new_sol


def fitness(y_true, proba):

    score = metrics.roc_auc_score(y_true, proba)
    # score = precision_recall_auc_score(y_true, proba)

    return score

def bin2cont(list_bin, fact):
    
    size = len(list_bin)
    param = 0
    
    for i in range(size):
        param = param + list_bin[i]*(2**i)
        
    param = fact*param/((2**size))
    
    return param

def crear_soluciones(num_sols, min_lay, max_lay, min_neu, max_neu, min_alpha, 
                     max_alpha, min_lr, max_lr, bin_cont = False):

    soluciones = []

    for sol in range(num_sols):
        ### Genera cantidad de neuronas y capas de forma aleatoria
        layer = []
        num_layers = random.randint(min_lay, max_lay)

        for i in range(num_layers):
            layer.append(random.randint(min_neu, max_neu))

        layer = tuple(layer)

        ### Selecciona de forma aleatoria la funcion de activacion
        activation = random.choice(["identity", "logistic"])
        
        if bin_cont:
            
            size = 15
            
            alpha_bin = (np.random.random(size)>0.5).astype(int)
            lr_bin = (np.random.random(size)>0.5).astype(int)
            
            alpha = bin2cont(alpha_bin, max_alpha)
            lr = bin2cont(lr_bin, max_lr)

    
            solucion = {
                "layer_sizes": layer,
                "activation": activation,
                "alpha": alpha,
                "alpha_bin": alpha_bin,
                "learning_rate_init": lr,
                "learning_rate_init_bin": lr_bin
            }
            
        else:
            ### Selecciona de forma aleatoria el alpha
            alpha = random.uniform(min_alpha, max_alpha)
    
            ### Selecciona de forma aleatoria un learning rate
    
            lr = random.uniform(min_lr, max_lr)
    
            solucion = {
                "layer_sizes": layer,
                "activation": activation,
                "alpha": alpha,
                "learning_rate_init": lr,
            }
        soluciones.append(solucion)

    return soluciones


def change_layer(hijo, min_lay, max_lay, min_neu, max_neu):

    cant_layers = random.randint(min_lay, max_lay)

    if len(hijo["layer_sizes"]) < cant_layers:
        layer = list(hijo["layer_sizes"])

        for i in range(cant_layers - len(hijo["layer_sizes"])):
            layer.append(random.randint(min_neu, max_neu))
    else:
        layer = list(hijo["layer_sizes"])
        for i in range(cant_layers):
            layer[i] = random.randint(min_neu, max_neu)

    return layer


def change_activation(hijo):

    opciones = ["identity", "logistic"]
    cambios = []

    for op in opciones:
        if op != hijo["activation"]:
            cambios.append(op)

    sel = random.choice(cambios)

    return sel


def change_alpha(hijo, min_alpha, max_alpha, bin_cont = False):
    
    if bin_cont:
        alpha_list_aux = hijo["alpha_bin"].copy()
        cell2change = random.randint(0,len(alpha_list_aux)-1)

        if alpha_list_aux[cell2change]==0:
            alpha_list_aux[cell2change]=1
        else:
            alpha_list_aux[cell2change]=0
        
        new_alpha = bin2cont(alpha_list_aux, 10)
        
    else:
        new_alpha = hijo["alpha"] + random.uniform(-hijo["alpha"], hijo["alpha"])
        new_alpha = min(new_alpha, max_alpha)
        new_alpha = max(new_alpha, min_alpha)
        alpha_list_aux = []

    return new_alpha, alpha_list_aux


def change_learningRate(hijo, min_lr, max_lr, bin_cont = False):
    
    if bin_cont:
        lr_list_aux = hijo["learning_rate_init_bin"].copy()
        cell2change = random.randint(0,len(lr_list_aux)-1)

        if lr_list_aux[cell2change]==0:
            lr_list_aux[cell2change]=1
        else:
            lr_list_aux[cell2change]=0
        
        new_lr = bin2cont(lr_list_aux, 1)
        
    else:
        new_lr = hijo["learning_rate_init"] + random.uniform(-hijo["learning_rate_init"], hijo["learning_rate_init"])
        new_lr = min(new_lr, max_lr)
        new_lr = max(new_lr, min_lr)
        lr_list_aux = []

    return new_lr, lr_list_aux


### Esta funcion realiza el entrenamiento de una red neuronal con unos parametros
### de entrada. Realiza un proceso de kfolds en la validacion del modelo
def entrenar_NN(X, Y, X_val, Y_val, k, params):

    NN = MLPClassifier(
        solver="adam",
        max_iter=300,
        random_state=42,
        beta_1=0.9,
        beta_2=0.999,
        n_iter_no_change=10,
        hidden_layer_sizes=params["layer_sizes"],
        activation=params["activation"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
    )

    best_score = 0
    best_mod = None
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):

        NN_mod = clone(NN)

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = Y.iloc[train_index]
        y_test = Y.iloc[test_index]

        NN_mod.fit(X_train, y_train)
        predictions = NN_mod.predict_proba(X_val)[:, 1]

        actuals = Y_val
        score = fitness(actuals, predictions)

        if score > best_score:
            # print(score)
            best_score = score
            best_mod = NN_mod

    return best_mod, best_score


def binarize_parameter(val, val_list):
    
    bin_sol = np.zeros(len(val_list))
    
    for i in range(1,len(val_list)):
        if (val_list[i-1] <= val) & (val <= val_list[i]):
            bin_sol[i] = 1
            break
        
    return bin_sol


def create_neighborhood(bin_var1, bin_var2, n_size, bin_cont = False):
    
    if bin_cont:
        index_v1 = []
        for i in range(len(bin_var1)):
            aux = bin_var1.copy()
            
            if bin_var1[i]==0:
                aux[i] = 1
            else:
                aux[i] = 0

            index_v1.append(aux)  
        
        index_v2 = []
        for i in range(len(bin_var2)):
            aux = bin_var2.copy()
            
            if bin_var2[i]==0:
                aux[i] = 1
            else:
                aux[i] = 0

            index_v2.append(aux)        
        
        ### creacion de vecindario
        
        ### se toma una muestra de forma que el vecindario sea mas pequeno
        ### y asi ayudar al tiempo de computo
        index_v1_sample = random.sample(index_v1, n_size)
        index_v2_sample = random.sample(index_v2, n_size)
        
        neighborhood = []
        for v1 in index_v1_sample:
            for v2 in index_v2_sample:
                neighborhood.append((v1,v2))
    else:
        pos_v1 = np.where(bin_var1==1)[0][0]
        pos_v2 = np.where(bin_var2==1)[0][0]
        
        N = math.floor(n_size/2)
        
        ### creacion de indices de los vecinos
        index_v1_aux = list(range(pos_v1-N,pos_v1+N+2))
        
        ### truncamiento de los indices
        index_v1 = []
        for v in index_v1_aux:
            if v < 0:
                index_v1.append(len(bin_var1)+v)
            elif v>= len(bin_var1):
                index_v1.append(v - len(bin_var1))
            else:
                index_v1.append(v)
    
    
        ### creacion de indices de los vecinos
        index_v2_aux = list(range(pos_v2-N,pos_v2+N+2))
        
        ### truncamiento de los indices
        index_v2 = []
        for v in index_v2_aux:
            if v < 0:
                index_v2.append(len(bin_var2)+v)
            elif v>= len(bin_var2):
                index_v2.append(v - len(bin_var2))
            else:
                index_v2.append(v)
    
    ### Para moverse unicamente al lado derecho    
    #    if pos_v1+n_size <len(bin_var1):
    #        index_v1 = list(range(pos_v1,pos_v1+n_size+1))
    #    else:
    #        val1 = len(bin_var1) - pos_v1
    #        range_1 = list(range(pos_v1, pos_v1+val1))
    #        range_2 = list(range(1+n_size-val1))
    #        index_v1 = [*range_1, *range_2]
    #
    #    if pos_v2+n_size <len(bin_var2):
    #        index_v2 = list(range(pos_v2,pos_v2+n_size+1))
    #    else:
    #        val2 = len(bin_var2) - pos_v2
    #        range_1 = list(range(pos_v2, pos_v2+val2))
    #        range_2 = list(range(1+n_size-val2))
    #        index_v2 = [*range_1, *range_2]
            
        ### Crear el vecindario
        neighborhood = []
        for i in index_v1:
            sol_var1 = np.zeros(len(bin_var1))
            sol_var1[i] = 1
            for j in index_v2:
                sol_var2 = np.zeros(len(bin_var2))
                sol_var2[j] = 1
                
                neighborhood.append((sol_var1,sol_var2))
    
    return neighborhood
