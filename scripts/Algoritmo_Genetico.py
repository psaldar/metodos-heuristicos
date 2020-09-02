# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:08:51 2020

@author: Pablo Saldarriaga
"""
import os
import json
import time
import random
import funciones
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
import sklearn.metrics as metrics

# %%
# Define el archivo en el que se guararan los logs del codigo
import logging
from logging.handlers import RotatingFileHandler

file_name = "Algoritmo_genetico"
logger = logging.getLogger()
dir_log = f"../logs/{file_name}.log"

# Crea la carpeta de logs
if not os.path.isdir("../logs"):
    os.makedirs("../logs", exist_ok=True)

handler = RotatingFileHandler(dir_log, maxBytes=2000000, backupCount=10)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    handlers=[handler],
)
# %%
# Esta funcion obtiene la metrica para cada uno de los modelos entrenados
# con las diferentes soluciones


def evaluar_fitness_poblacion(X, Y, X_val, Y_val, population, k=3):

    poblacion = population.copy()
    fitness = []
    best_score = 0
    best_mod = None
    for i in range(len(poblacion)):

        if not "fitness" in poblacion[i].keys():
            _, score = funciones.entrenar_NN(X, Y, X_val, Y_val, k, poblacion[i])
            fitness.append(score)
            poblacion[i]["fitness"] = score
        else:
            score = poblacion[i]["fitness"]
            fitness.append(score)

        if score > best_score:
            best_score = score
            best_mod = poblacion[i]

    fitness = np.array(fitness)

    return poblacion, fitness, best_mod, best_score


def seleccion(poblacion):

    ### Seleccion por metodo de torneo
    ### Definicion de cantidad de individuos a participar en el torneo
    k = 5
    fitness = []
    participantes = random.sample(poblacion, k)

    for i in range(k):
        fitness.append(participantes[i]['fitness'])

    indices_best2worst = np.argsort(fitness)[::-1]

    ### Se seleccionan los dos mejores individuos acorde a su fitness
    padre_1 = participantes[indices_best2worst[0]]
    padre_2 = participantes[indices_best2worst[1]]

    return padre_1, padre_2


def get_bestPoblacion(fitness, poblacion, cant_seleccion):

    indices_best2worst = np.argsort(fitness)[::-1]
    # fitness_best2worst = fitness[indices_best2worst]

    poblacion_best2worst = []
    for ind in indices_best2worst:
        poblacion_best2worst.append(poblacion[ind])

    best_pop = poblacion_best2worst[:cant_seleccion]

    return best_pop


def crossover(X, Y, X_val, Y_val, k, padre_1a, padre_2a):
    ### Cruce por dos puntos

    padre_1 = padre_1a.copy()
    padre_2 = padre_2a.copy()

    array_p1 = funciones.sol2array(padre_1)
    array_p2 = funciones.sol2array(padre_2)

    array_cruce1 = np.zeros(len(array_p1))
    array_cruce2 = np.zeros(len(array_p1))

    ### Encuentro los dos puntos en los cuales se hace el cruce
    aa = -1
    bb = -1

    while (aa == bb) or (abs(aa - bb) < 2):
        aa = random.randint(1, len(array_p1) - 2)
        bb = random.randint(1, len(array_p1) - 2)

    a = min(aa, bb)
    b = max(aa, bb)

    for i in range(len(array_p1)):

        if i < a:
            array_cruce1[i] = array_p1[i]
            array_cruce2[i] = array_p2[i]

        elif i < b:
            array_cruce1[i] = array_p2[i]
            array_cruce2[i] = array_p1[i]

        else:
            array_cruce1[i] = array_p1[i]
            array_cruce2[i] = array_p2[i]

    hijo_1 = funciones.array2sol(array_cruce1)
    hijo_2 = funciones.array2sol(array_cruce2)

    _, score_h1 = funciones.entrenar_NN(X, Y, X_val, Y_val, k, hijo_1)
    _, score_h2 = funciones.entrenar_NN(X, Y, X_val, Y_val, k, hijo_2)

    if score_h1 >= score_h2:
        best_h = hijo_1
        hijo_1['fitness'] = score_h1
    else:
        best_h = hijo_2
        hijo_2['fitness'] = score_h2

    return best_h


def mutacion(hijo_a, min_lay, max_lay, min_neu, max_neu, min_alpha, max_alpha, min_lr, max_lr, escenario=1):

    hijo = hijo_a.copy()
    ops = ["layer_sizes", "activation", "alpha", "learning_rate_init"]
    if escenario == 1:
        ### Este escenario selecciona de forma aleatoria uno de los 4 hiperparametros a variar
        gen = random.choice(ops)

        if gen == "layer_sizes":

            layer = funciones.change_layer(hijo, min_lay, max_lay, min_neu, max_neu)

            hijo["layer_sizes"] = tuple(layer)

        elif gen == "activation":

            hijo["activation"] = funciones.change_activation(hijo)

        elif gen == "alpha":
            hijo["alpha"] = funciones.change_alpha(hijo, min_alpha, max_alpha)

        else:
            hijo["learning_rate_init"] = funciones.change_learningRate(hijo, min_lr, max_lr)
    elif escenario == 2:
        ### Este escenario le asigna una probabilidad de mutacion a cada hiperparametro
        probabilidades = [0.7, 0.6, 0.4, 0.4]

        ### layers
        if random.random() < probabilidades[0]:

            layer = funciones.change_layer(hijo, min_lay, max_lay, min_neu, max_neu)

            hijo["layer_sizes"] = tuple(layer)

        ### activation
        if random.random() < probabilidades[1]:

            hijo["activation"] = funciones.change_activation(hijo)

        ### alpha
        if random.random() < probabilidades[2]:
            hijo["alpha"] = funciones.change_alpha(hijo, min_alpha, max_alpha)

        ### Learning rate
        if random.random() < probabilidades[3]:
            hijo["learning_rate_init"] = funciones.change_learningRate(hijo, min_lr, max_lr)

    elif escenario == 3:
        ### Este escenario selecciona de forma aleatoria uno de los 4 hiperparametros a variar.
        ### la diferencia con el escenario 1, es que  se le asignan pesos a cada uno de los hiperparametros
        ### por lo que la seleccion no es uniforme

        gen_n = np.random.choice([0, 1, 2, 3], 1, p=[0.6, 0.2, 0.1, 0.1], replace=False)
        gen = ops[gen_n[0]]
        if gen == "layer_sizes":

            layer = funciones.change_layer(hijo, min_lay, max_lay, min_neu, max_neu)

            hijo["layer_sizes"] = tuple(layer)

        elif gen == "activation":

            hijo["activation"] = funciones.change_activation(hijo)

        elif gen == "alpha":
            hijo["alpha"] = funciones.change_alpha(hijo, min_alpha, max_alpha)

        else:
            hijo["learning_rate_init"] = funciones.change_learningRate(hijo, min_lr, max_lr)

    return hijo


def update_poblacion(X_train, Y_train, X_val, Y_val, poblacion_prev, fitness, hijos_prev, k):

    poblacion = poblacion_prev.copy()
    hijos = hijos_prev.copy()

    cant_seleccion = len(poblacion) - len(hijos)

    best_pop = get_bestPoblacion(fitness, poblacion, cant_seleccion)

    new_poblacion = [*best_pop, *hijos]
    new_poblacion, fitness, best_mod, best_score = evaluar_fitness_poblacion(X_train,
                                                                             Y_train,
                                                                             X_val,
                                                                             Y_val,
                                                                             new_poblacion,
                                                                             k=k)

    return new_poblacion, fitness, best_mod, best_score


# %%


def AG(X_train,
       Y_train,
       X_val,
       Y_val,
       k=3,
       num_generacion=5,
       population_size=10,
       prob_seleccion=0.5,
       prob_mutacion=0.2,
       escenario=1,
       iteracion = 0,
       desc = 'run'):

    time_ini = time.time()

    logger.info("***" * 20)
    logger.info("INICIA ALGORITMO GENETICO")
    logger.info("***" * 20)

    # Fijar semillas
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # Definicion de parametros del AG
    cant_seleccion = int(population_size * prob_seleccion)

    # Creacion de la poblacion
    min_lay = 1
    max_lay = 4
    min_neu = 1
    max_neu = 128
    min_alpha = 0.0001
    max_alpha = 10
    min_lr = 0.001
    max_lr = 1

    historia_score = []

    poblacion = funciones.crear_soluciones(
        population_size,
        min_lay,
        max_lay,
        min_neu,
        max_neu,
        min_alpha,
        max_alpha,
        min_lr,
        max_lr,
    )
    poblacion, fitness, best_mod, best_score = evaluar_fitness_poblacion(X_train, Y_train, X_val, Y_val, poblacion, k=k)

    historia_score.append(best_score)

    for i in tqdm(range(num_generacion)):
        logger.info(f'generacion {i} de {num_generacion}')
        hijos = []
        for j in range(population_size - cant_seleccion):

            padre_1, padre_2 = seleccion(poblacion)
            hijo = crossover(X_train, Y_train, X_val, Y_val, k, padre_1, padre_2)

            if random.random() < prob_mutacion:
                hijo = mutacion(hijo,
                                min_lay,
                                max_lay,
                                min_neu,
                                max_neu,
                                min_alpha,
                                max_alpha,
                                min_lr,
                                max_lr,
                                escenario=escenario)

            hijos.append(hijo)

        poblacion, fitness, mod, score = update_poblacion(X_train,
                                                          Y_train,
                                                          X_val,
                                                          Y_val,
                                                          poblacion,
                                                          fitness,
                                                          hijos,
                                                          k=k)

        historia_score.append(score)

        if score > best_score:
            best_score = score
            best_mod = mod

    bes_mod_entrenado, _ = funciones.entrenar_NN(X_train, Y_train, X_val, Y_val, k, best_mod)

    time_exec = time.time() - time_ini

    ### Guardo resultados
    res = {
        'population_size': population_size,
        'num_generaciones': num_generacion,
        'prob_selecicon': prob_seleccion,
        'prob_mutacion': prob_mutacion,
        'best_score': best_score,
        'best_mod': best_mod,
        'execution_time_mins': time_exec,
        'history': historia_score
    }

    now = dt.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    with open(f'../data/output/output_{escenario}_{iteracion}_{now}_{desc}.json', 'w') as json_file:
        json.dump(res, json_file)

    logger.info("***" * 20)
    logger.info(f"FIN ALGORITMO GENETICO: tiempo {round(time_exec,4)} mins")
    logger.info("***" * 20)

    logger.info("La mejor solucion fue")
    logger.info(str(best_mod))

    return best_mod, best_score, bes_mod_entrenado, historia_score


# %%
# if __name__ == "__main__":

#     logger.info("CARGA DE DATOS")
#     # Carga la informacion para entrenar el modelo
#     train = pd.read_csv("../data/train_z.csv").drop(columns=["TW", "BARRIO"])
#     validation = pd.read_csv("../data/validation_z.csv").drop(columns=["TW", "BARRIO"])
#     test = pd.read_csv("../data/test_z.csv").drop(columns=["TW", "BARRIO"])

#     # Divide conjunto de datos en entrenamiento y en prueba
#     X_train = train.drop(columns="Accidente")
#     Y_train = train["Accidente"]

#     X_val = validation.drop(columns="Accidente")
#     Y_val = validation["Accidente"]

#     X_test = test.drop(columns="Accidente")
#     Y_test = test["Accidente"]

#     # Realiza la ejecucion del algoritmo genetico
#     best_mod, best_score, bes_mod_entrenado, historia_score = AG(X_train,
#                                                                  Y_train,
#                                                                  X_val,
#                                                                  Y_val,
#                                                                  num_generacion=5,
#                                                                  population_size=10,
#                                                                  prob_seleccion=0.5,
#                                                                  prob_mutacion=0.2,
#                                                                  escenario=1)

#     # realiza las predicciones en el conjutno de prueba
#     proba_test = bes_mod_entrenado.predict_proba(X_test)[:, 1]
#     preds_test = bes_mod_entrenado.predict(X_test)

#     # Calcula las metricas del modelo en el conjunto de prueba
#     PR_auc = funciones.precision_recall_auc_score(Y_test, proba_test)
#     ROC_auc = metrics.roc_auc_score(Y_test, proba_test)

#     bAccuracy = metrics.balanced_accuracy_score(Y_test, preds_test)
#     precision = metrics.precision_score(Y_test, preds_test)
#     recall = metrics.recall_score(Y_test, preds_test)
#     fscore = metrics.f1_score(Y_test, preds_test)

#     logger.info("Desempeño en el conjunto de prueba:")
#     logger.info(f"PR-AUC:{PR_auc}")
#     logger.info(f"ROC-AUC:{ROC_auc}")
#     logger.info(f"Balanced accuracy:{bAccuracy}")
#     logger.info(f"Precision:{precision}")
#     logger.info(f"Recall:{recall}")
#     logger.info(f"F1 Score:{fscore}")

# %%