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


def crossover(X, Y, X_val, Y_val, k, padre_1a, padre_2a, bin_cont = False):
    ### Cruce por dos puntos

    padre_1 = padre_1a.copy()
    padre_2 = padre_2a.copy()

    array_p1 = funciones.sol2array(padre_1, bin_cont)
    array_p2 = funciones.sol2array(padre_2, bin_cont)

    array_cruce1 = list(np.zeros(len(array_p1)))
    array_cruce2 = list(np.zeros(len(array_p1)))

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

    hijo_1 = funciones.array2sol(array_cruce1, bin_cont)
    hijo_2 = funciones.array2sol(array_cruce2, bin_cont)

    _, score_h1 = funciones.entrenar_NN(X, Y, X_val, Y_val, k, hijo_1)
    _, score_h2 = funciones.entrenar_NN(X, Y, X_val, Y_val, k, hijo_2)

    if score_h1 >= score_h2:
        best_h = hijo_1
        best_h['fitness'] = score_h1
    else:
        best_h = hijo_2
        best_h['fitness'] = score_h2

    return best_h


def mutacion(hijo_a, min_lay, max_lay, min_neu, max_neu, min_alpha, max_alpha, 
             min_lr, max_lr, escenario=1,bin_cont = False):

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
            hijo["alpha"], hijo["alpha_bin"] = funciones.change_alpha(hijo, min_alpha, max_alpha, bin_cont = bin_cont)

        else:
            hijo["learning_rate_init"], hijo["learning_rate_init_bin"] = funciones.change_learningRate(hijo, min_lr, max_lr, bin_cont = bin_cont)
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
            hijo["alpha"], hijo["alpha_bin"] = funciones.change_alpha(hijo, min_alpha, max_alpha, bin_cont = bin_cont)

        ### Learning rate
        if random.random() < probabilidades[3]:
            hijo["learning_rate_init"], hijo["learning_rate_init_bin"] = funciones.change_learningRate(hijo, min_lr, max_lr, bin_cont = bin_cont)

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
            hijo["alpha"], hijo["alpha_bin"] = funciones.change_alpha(hijo, min_alpha, max_alpha, bin_cont = bin_cont)

        else:
            hijo["learning_rate_init"], hijo["learning_rate_init_bin"] = funciones.change_learningRate(hijo, min_lr, max_lr, bin_cont = bin_cont)

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


def busqueda_exahustiva(X_train, Y_train, X_val, Y_val, k, best_mod):
    logger.info('Inicia busqueda exahustiva')
    best_sol = best_mod.copy()
    
    sol_aux = best_mod.copy()
    for f_activation in ["identity", "logistic", "tanh", "relu"]:
        
        for n_layers in range(1,5):
            
            if n_layers <= len(sol_aux['layer_sizes']):
                new_layers = sol_aux['layer_sizes'][:n_layers]
            else:
                ones = []
                cant_ones = n_layers - len(sol_aux['layer_sizes'])
                for _ in range(cant_ones):
                    ones.append(1)
                
                new_layers = tuple([*list(sol_aux['layer_sizes']), *ones])
                
            new_sol = best_mod.copy()
            new_sol['layer_sizes'] = new_layers
            new_sol['activation'] = f_activation
            
            _, score = funciones.entrenar_NN(X_train, Y_train, X_val, Y_val, k, new_sol)
            
            new_sol["fitness"] = score   
            
            if best_sol['fitness']< new_sol["fitness"]:
                best_sol = new_sol.copy()
                logger.info(f'Mejora la solucion con la busqueda exahustiva: {best_sol}')
            
        
    return best_sol

def busqueda_local(X_train, Y_train, X_val, Y_val, k, min_alpha, max_alpha,
                                min_lr, max_lr, best_mod, vec_size =5, 
                                bin_cont = False):
    ### inicializo soluciones a considerar
    best_sol = best_mod.copy()    
       
    
    lr = best_sol['learning_rate_init']
    alpha = best_sol['alpha']
    
    ### discretizo los posibles valores de learning_rate y alpha
    step_size = 20
    h_lr = (max_lr-min_lr)/step_size
    h_alpha = (max_alpha-min_alpha)/step_size
    
    lr_list = []
    alpha_list = []
    for i in range(step_size+1):
        lr_list.append(min_lr+h_lr*i)
        alpha_list.append(min_alpha+h_alpha*i)
    
    lr_list = np.array(lr_list)
    alpha_list = np.array(alpha_list)
    
    ### obtengo binarizacion de las dos variables continuas
    bin_lr = funciones.binarize_parameter(lr, lr_list)
    bin_alpha = funciones.binarize_parameter(alpha, alpha_list)
    

    if bin_cont:
        logger.info('Entra a bin cont')
        ### creacion del vecindario. 
        vecindario = funciones.create_neighborhood(best_mod['learning_rate_init_bin'], best_mod['alpha_bin'], vec_size, bin_cont)
        ### iterar sobre el vecindario
        for vecino in vecindario:
            lr_vecino = vecino[0]
            alpha_vecino = vecino[1]
        
            new_sol = best_mod.copy()
            new_sol['learning_rate_init_bin'] = lr_vecino
            new_sol['learning_rate_init'] = funciones.bin2cont(lr_vecino, 1)
            new_sol['alpha_bin'] = alpha_vecino
            new_sol['alpha'] = funciones.bin2cont(alpha_vecino, 10)   
            
            ### obtengo el fitness del vecino en la iteracion
            _, score = funciones.entrenar_NN(X_train, Y_train, X_val, Y_val, k, new_sol)
            new_sol["fitness"] = score  
            
            ### conserva el primer vecino que mejore la FO
            if best_sol['fitness']< new_sol["fitness"]:
                best_sol = new_sol.copy()
                logger.info(f'Mejora la solucion con la busqueda local: {best_sol}')
                print('Mejora')
                return best_sol        
    else:
        vecindario = funciones.create_neighborhood(bin_lr, bin_alpha, vec_size, bin_cont)
        ### iterar sobre el vecindario
        for vecino in vecindario:
            lr_vecino = vecino[0]
            alpha_vecino = vecino[1]
        
            new_sol = best_mod.copy()
            new_sol['learning_rate_init'] = lr_list[lr_vecino==1][0]
            new_sol['alpha'] = alpha_list[alpha_vecino==1][0]   
            
            ### obtengo el fitness del vecino en la iteracion
            _, score = funciones.entrenar_NN(X_train, Y_train, X_val, Y_val, k, new_sol)
            new_sol["fitness"] = score  
            
            ### conserva el primer vecino que mejore la FO
            if best_sol['fitness']< new_sol["fitness"]:
                best_sol = new_sol.copy()
                logger.info(f'Mejora la solucion con la busqueda local: {best_sol}')
                print('Mejora')
                return best_sol
        
    logger.info('NO HAY MEJORA EN EL VECINDARIO')
    return best_sol

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
       desc = 'run',
       BH = False,
       LS = False,
       bin_cont = False,
       vec_size = 5):

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
        bin_cont = bin_cont)
    
    poblacion, fitness, best_mod, best_score = evaluar_fitness_poblacion(X_train, Y_train, X_val, Y_val, poblacion, k=k)

    historia_score.append(best_score)
    last_sol = ''
    for i in tqdm(range(num_generacion)):
        logger.info(f'generacion {i} de {num_generacion}')
        hijos = []
        for j in range(population_size - cant_seleccion):

            padre_1, padre_2 = seleccion(poblacion)
            hijo = crossover(X_train, Y_train, X_val, Y_val, k, padre_1, padre_2, bin_cont = bin_cont)

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
                                escenario=escenario,
                                bin_cont = bin_cont)

            hijos.append(hijo)

        poblacion, fitness, mod, score = update_poblacion(X_train,
                                                          Y_train,
                                                          X_val,
                                                          Y_val,
                                                          poblacion,
                                                          fitness,
                                                          hijos,
                                                          k=k)

        
        
        if score > best_score:
            best_score = score
            best_mod = mod.copy()
            
        ### BUSQUEDA EXAHUSTIVA
        if BH:
            if not last_sol==str(best_mod):
                logger.info(f'Inicia busqueda exahustiva en la generacion al final de la generacion {i}')
                ### consideramos una busqueda exahustiva en la mejor solucion encontrada
                best_sol = busqueda_exahustiva(X_train, Y_train, X_val, Y_val, k, best_mod)
                
                last_sol = str(best_mod.copy())
                last_mod = best_mod.copy()
                ### actualiza la mejor solucion enconrada
                if best_sol['fitness'] > best_score:                    
                    
                    logger.info('Encuentra mejor solucion en busqueda exahustiva')
                    best_score = best_sol['fitness']
                    best_mod = best_sol.copy()    
                    
                    logger.info('Agrega a la poblacion el mejor individuo de la BH')
                    ### ingresamos a la poblacion el mejor individuo encontrado
                    if not best_mod in poblacion:
                        logger.info(f'AGREGADO:{best_mod}')
                        print(f'AGREGADO:{best_mod}')
                        poblacion[poblacion.index(last_mod)] = best_mod.copy()

            else:
                logger.info(f"YA SE EVALUO LA SOLUCION")
                logger.info(f"ANTES: {last_mod}")
                logger.info(f"AHORA: {best_mod}")
        ### BUSQUEDA LOCAL
        if LS:
            if not last_sol==str(best_mod):
                logger.info(f'Inicia busqueda local en la generacion al final de la generacion {i}')
                ### consideramos una busqueda exahustiva en la mejor solucion encontrada
                best_sol = busqueda_local(X_train, Y_train, X_val, Y_val, k, 
                                          min_alpha, max_alpha, min_lr, max_lr, 
                                          best_mod, vec_size = vec_size, bin_cont = bin_cont)
                
                last_sol = str(best_mod.copy())
                last_mod = best_mod.copy()
                ### actualiza la mejor solucion enconrada
                if best_sol['fitness'] > best_score:
                    
                    logger.info('Encuentra mejor solucion en busqueda local')
                    best_score = best_sol['fitness']
                    best_mod = best_sol.copy()    
                    
                    logger.info('Agrega a la poblacion el mejor individuo de LS')
                    if not best_mod in poblacion:
                        logger.info(f'AGREGADO:{best_mod}')
                        print(f'AGREGADO:{best_mod}')
                        ### ingresamos a la poblacion el mejor individuo encontrado
                        poblacion[poblacion.index(last_mod)] = best_mod.copy()
    #                    poblacion, fitness, mod, score = update_poblacion(X_train,
    #                                                                      Y_train,
    #                                                                      X_val,
    #                                                                      Y_val,
    #                                                                      poblacion,
    #                                                                      fitness,
    #                                                                      [best_mod],
    #                                                                      k=k)
            else:
                logger.info(f"YA SE EVALUO LA SOLUCION")
                logger.info(f"ANTES: {last_mod}")
                logger.info(f"AHORA: {best_mod}")
                
        historia_score.append(best_score)
        logger.info(f'Historia: {historia_score}')

    bes_mod_entrenado, _ = funciones.entrenar_NN(X_train, Y_train, X_val, Y_val, k, best_mod)

    time_exec = time.time() - time_ini

    ### Guardo resultados
    if bin_cont:
        best_mod['learning_rate_init_bin'] = str(list(best_mod['learning_rate_init_bin']))
        best_mod['alpha_bin'] = str(list(best_mod['alpha_bin']))
    else:
        best_mod['learning_rate_init_bin'] = str([])
        best_mod['alpha_bin'] = str([])        
    
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


## %%
#if __name__ == "__main__":
#
#    logger.info("CARGA DE DATOS")
#    # Carga la informacion para entrenar el modelo
#    train = pd.read_csv("../data/train_z.csv").drop(columns=["TW", "BARRIO"])
#    validation = pd.read_csv("../data/validation_z.csv").drop(columns=["TW", "BARRIO"])
#    test = pd.read_csv("../data/test_z.csv").drop(columns=["TW", "BARRIO"])
#
#    # Divide conjunto de datos en entrenamiento y en prueba
#    X_train = train.drop(columns="Accidente")
#    Y_train = train["Accidente"]
#
#    X_val = validation.drop(columns="Accidente")
#    Y_val = validation["Accidente"]
#
#    X_test = test.drop(columns="Accidente")
#    Y_test = test["Accidente"]
#
#    # Realiza la ejecucion del algoritmo genetico
#    best_mod, best_score, bes_mod_entrenado, historia_score = AG(X_train,
#                                                                  Y_train,
#                                                                  X_val,
#                                                                  Y_val,
#                                                                  num_generacion=5,
#                                                                  population_size=10,
#                                                                  prob_seleccion=0.5,
#                                                                  prob_mutacion=0.2,
#                                                                  escenario=1,
#                                                                  BH = False,
#                                                                  LS = True,
#                                                                  bin_cont = True)
#
#    # realiza las predicciones en el conjutno de prueba
#    proba_test = bes_mod_entrenado.predict_proba(X_test)[:, 1]
#    preds_test = bes_mod_entrenado.predict(X_test)
#
#    # Calcula las metricas del modelo en el conjunto de prueba
#    PR_auc = funciones.precision_recall_auc_score(Y_test, proba_test)
#    ROC_auc = metrics.roc_auc_score(Y_test, proba_test)
#
#    bAccuracy = metrics.balanced_accuracy_score(Y_test, preds_test)
#    precision = metrics.precision_score(Y_test, preds_test)
#    recall = metrics.recall_score(Y_test, preds_test)
#    fscore = metrics.f1_score(Y_test, preds_test)
#
#    logger.info("Desempeño en el conjunto de prueba:")
#    logger.info(f"PR-AUC:{PR_auc}")
#    logger.info(f"ROC-AUC:{ROC_auc}")
#    logger.info(f"Balanced accuracy:{bAccuracy}")
#    logger.info(f"Precision:{precision}")
#    logger.info(f"Recall:{recall}")
#    logger.info(f"F1 Score:{fscore}")

# %%
