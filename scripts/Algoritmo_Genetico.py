# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:08:51 2020

@author: Pablo Saldarriaga
"""
import os
import json
import random
import funciones
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
#%%
#Define el archivo en el que se guararan los logs del codigo
import logging
from logging.handlers import RotatingFileHandler

file_name = 'Algoritmo_genetico'
logger = logging.getLogger()
dir_log = f'../logs/{file_name}.log'

### Crea la carpeta de logs
if not os.path.isdir('../logs'):
    os.makedirs('../logs', exist_ok=True) 

handler = RotatingFileHandler(dir_log, maxBytes=2000000, backupCount=10)
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
                    handlers = [handler])
#%%
### Esta funcion obtiene la metrica para cada uno de los modelos entrenados
### con las diferentes soluciones
def evaluar_fitness_poblacion(X, Y, X_val, Y_val, poblacion):
    
    fitness = []
    best_score = 0
    best_mod = None
    for i in range(len(poblacion)):
        
        if not 'fitness' in poblacion[i].keys():
            mod, score = funciones.entrenar_NN(X, Y,X_val, Y_val, 3, poblacion[i])
            fitness.append(score)
            poblacion[i]['fitness'] = score
        else:
            score = poblacion[i]['fitness']
            fitness.append(score)
        
        if score > best_score:
            best_score = score
            best_mod = poblacion[i]
            
    fitness = np.array(fitness)
    
    return poblacion, fitness, best_mod, best_score

def seleccion(fitness, poblacion, cant_seleccion):

    indices_best2worst = np.argsort(fitness)[::-1]
    fitness_best2worst = fitness[indices_best2worst]
    
    poblacion_best2worst = []
    for ind in indices_best2worst:
        poblacion_best2worst.append(poblacion[ind])   
    
    
    best_pop = poblacion_best2worst[:cant_seleccion]
    
    ### ACTUALMENTE LA SELECCION ESTA ALEATORIA, HAY QUE CAMBIARLA
    padre_1, padre_2 = random.sample(best_pop, 2)
    
    return padre_1, padre_2, best_pop
    

def crossover(padre_1, padre_2):
    
    ### Cruce de cantidad de neuronas y capas
    min_l = min(len(padre_1['layer_sizes']),len(padre_2['layer_sizes']))
    max_l = max(len(padre_1['layer_sizes']),len(padre_2['layer_sizes']))
    l_cross = random.randint(min_l, max_l)
    
    min_n = min(min(padre_1['layer_sizes']),min(padre_2['layer_sizes']))
    max_n = max(max(padre_1['layer_sizes']),max(padre_2['layer_sizes']))
    
    neuronas = []
    for i in range(l_cross):
        neuronas.append(random.randint(min_n, max_n))
    
    new_layer = tuple(neuronas)
    
    ### cruce de funciones de activavion
    new_activation = random.choice([padre_1['activation'],padre_2['activation']])
    
    ### cruce alpha
    min_alpha = min(padre_1['alpha'],padre_2['alpha'])
    max_alpha = max(padre_1['alpha'],padre_2['alpha'])
    
    new_alpha = random.uniform(min_alpha, max_alpha)
    
    ### Cruce de init learning rate
    min_lr = min(padre_1['learning_rate_init'],padre_2['learning_rate_init'])
    max_lr = max(padre_1['learning_rate_init'],padre_2['learning_rate_init'])
    
    new_lr = random.uniform(min_lr, max_lr)
    
    cruce = {'layer_sizes': new_layer,
             'activation': new_activation,
             'alpha': new_alpha,
             'learning_rate_init': new_lr
             }
    
    return cruce

def mutacion(hijo, min_lay, max_lay, min_neu, max_neu):
    
    gen = random.choice(['layer_sizes','activation', 'alpha', 'learning_rate_init'])
    
    if gen == 'layer_sizes':
        cant_layers = random.randint(min_lay, max_lay)
        
        if len(hijo['layer_sizes']) < cant_layers:
            layer = list(hijo['layer_sizes'])
            
            for i in range(cant_layers-len(hijo['layer_sizes'])):
                layer.append(random.randint(min_neu, max_neu))
        else:
            layer = list(hijo['layer_sizes'])
            for i in range(cant_layers):
                layer[i] = random.randint(min_neu, max_neu)
                
        hijo['layer_sizes'] = tuple(layer)
        
    elif gen == 'activation': 
        
        opciones = ['identity', 'logistic', 'tanh', 'relu']
        cambios = []
        
        for op in opciones:
            if op != hijo['activation']:
                cambios.append(op)
                
        hijo['activation'] = random.choice(cambios)
        
    elif gen == 'alpha':
        hijo['alpha'] = hijo['alpha'] +random.uniform(-hijo['alpha'], hijo['alpha'])
        
    else: 
        hijo['learning_rate_init'] = hijo['learning_rate_init'] +random.uniform(-hijo['learning_rate_init'], hijo['learning_rate_init'])
        
    
    return hijo


def update_poblacion(X_train, Y_train, X_val, Y_val, poblacion, fitness, hijos):
    
    cant_seleccion = len(poblacion) - len(hijos)
    
    _, _, best_pop = seleccion(fitness, poblacion, cant_seleccion)
    
    new_poblacion = [*best_pop, *hijos]
    new_poblacion, fitness, best_mod, best_score = evaluar_fitness_poblacion(X_train, Y_train, X_val, Y_val, new_poblacion)
    
    return new_poblacion, fitness, best_mod, best_score

#%%
def AG(X_train, Y_train, X_val, Y_val, num_generacion = 5, 
       population_size = 10, prob_seleccion = 0.5, prob_mutacion = 0.2):
    
    logger.info('***'*20)
    logger.info('INICIA ALGORITMO GENETICO')
    logger.info('***'*20)
    
    ### Fijar semillas
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    
    ### Definicion de parametros del AG    
    cant_seleccion = int(population_size*prob_seleccion)
    
    ### Creacion de la poblacion
    min_lay = 1
    max_lay = 4
    min_neu = 1
    max_neu = 128
    min_alpha = 0.0001
    max_alpha = 10
    min_lr = 0.001
    max_l = 1
    
    historia_score = []
    
    poblacion = funciones.crear_soluciones(population_size, min_lay, max_lay, min_neu, max_neu, min_alpha, max_alpha, min_lr, max_l)
    poblacion, fitness, best_mod, best_score = evaluar_fitness_poblacion(X_train, Y_train, X_val, Y_val, poblacion)
    
    historia_score.append(best_score)
 
    for i in range(num_generacion):
        hijos = []
        for j in range(population_size - cant_seleccion):
            
            padre_1, padre_2, _ = seleccion(fitness, poblacion, cant_seleccion)
            hijo = crossover(padre_1, padre_2)
            
            if random.random() < prob_mutacion:
                hijo = mutacion(hijo, min_lay, max_lay, min_neu, max_neu)
                
            hijos.append(hijo)
            
        poblacion, fitness, mod, score = update_poblacion(X_train, Y_train, X_val, Y_val, poblacion, fitness, hijos)
        
        historia_score.append(score)
        
        if score > best_score:
            best_score = score
            best_mod = mod
    
    bes_mod_entrenado, _ = funciones.entrenar_NN(X_train, Y_train, X_val, Y_val, 3, best_mod)
    
    return best_mod, best_score, bes_mod_entrenado, historia_score



#%%
if __name__ == '__main__':
    
    logger.info('CARGA DE DATOS')
    ### Carga la informacion para entrenar el modelo
    train = pd.read_csv('../data/train_z.csv').drop(columns = ['TW','BARRIO'])
    validation = pd.read_csv('../data/validation_z.csv').drop(columns = ['TW','BARRIO'])
    test = pd.read_csv('../data/test_z.csv').drop(columns = ['TW','BARRIO'])
    
    ### Divide conjunto de datos en entrenamiento y en prueba
    X_train = train.drop(columns = 'Accidente')
    Y_train = train['Accidente']
    
    X_val = validation.drop(columns = 'Accidente')
    Y_val = validation['Accidente']
    
    X_test = test.drop(columns = 'Accidente')
    Y_test = test['Accidente']
    
    ### Realiza la ejecucion del algoritmo genetico
    best_mod, best_score, bes_mod_entrenado, historia_score = AG(X_train, 
                                                                 Y_train,
                                                                 X_val, 
                                                                 Y_val, 
                                                                 num_generacion = 5, 
                                                                 population_size = 5, 
                                                                 prob_seleccion = 0.5, 
                                                                 prob_mutacion = 0.2)
    
    ### realiza las predicciones en el conjutno de prueba
    proba_test = bes_mod_entrenado.predict_proba(X_test)[:,1]
    preds_test = bes_mod_entrenado.predict(X_test)
    
    ### Calcula las metricas del modelo en el conjunto de prueba
    PR_auc = funciones.precision_recall_auc_score(Y_test, proba_test)
    ROC_auc = metrics.roc_auc_score(Y_test, proba_test)
    
    bAccuracy = metrics.balanced_accuracy_score(Y_test, preds_test)
    precision = metrics.precision_score(Y_test, preds_test)
    recall = metrics.recall_score(Y_test, preds_test)
    fscore = metrics.f1_score(Y_test, preds_test)
