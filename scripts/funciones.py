# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:12:15 2020

@author: Pablo Saldarriaga
"""
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


def sol2array(individuo):
    max_lay = 4
    sol = np.zeros(4 + max_lay)

    eq = {'activation': {"identity": 1, "logistic": 2, "tanh": 3, "relu": 4}}

    sol[0] = eq['activation'][individuo['activation']]
    sol[1] = individuo['alpha']
    sol[2] = individuo['learning_rate_init']
    sol[3] = len(individuo['layer_sizes'])

    for i in range(len(individuo['layer_sizes'])):
        sol[i + 4] = individuo['layer_sizes'][i]

    return sol


def array2sol(sol):

    eq = {'activation': {1: "identity", 2: "logistic", 3: "tanh", 4: "relu"}}

    activation = eq['activation'][int(sol[0])]
    alpha = sol[1]
    lr = sol[2]

    layers = []
    for i in range(4, len(sol)):
        if sol[i] != 0:
            layers.append(int(sol[i]))

    solucion = {
        "layer_sizes": tuple(layers),
        "activation": activation,
        "alpha": alpha,
        "learning_rate_init": lr,
    }

    return solucion


def fitness(y_true, proba):

    score = metrics.roc_auc_score(y_true, proba)
    # score = precision_recall_auc_score(y_true, proba)

    return score


def crear_soluciones(num_sols, min_lay, max_lay, min_neu, max_neu, min_alpha, max_alpha, min_lr, max_lr):

    soluciones = []

    for sol in range(num_sols):
        ### Genera cantidad de neuronas y capas de forma aleatoria
        layer = []
        num_layers = random.randint(min_lay, max_lay)

        for i in range(num_layers):
            layer.append(random.randint(min_neu, max_neu))

        layer = tuple(layer)

        ### Selecciona de forma aleatoria la funcion de activacion
        activation = random.choice(["identity", "logistic", "tanh", "relu"])

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

    opciones = ["identity", "logistic", "tanh", "relu"]
    cambios = []

    for op in opciones:
        if op != hijo["activation"]:
            cambios.append(op)

    sel = random.choice(cambios)

    return sel


def change_alpha(hijo, min_alpha, max_alpha):

    new_alpha = hijo["alpha"] + random.uniform(-hijo["alpha"], hijo["alpha"])
    new_alpha = min(new_alpha, max_alpha)
    new_alpha = max(new_alpha, min_alpha)

    return new_alpha


def change_learningRate(hijo, min_lr, max_lr):

    new_lr = hijo["learning_rate_init"] + random.uniform(-hijo["learning_rate_init"], hijo["learning_rate_init"])
    new_lr = min(new_lr, max_lr)
    new_lr = max(new_lr, min_lr)

    return new_lr


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
