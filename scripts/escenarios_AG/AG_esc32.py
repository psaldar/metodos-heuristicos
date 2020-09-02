# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:57:16 2020

@author: pasal
"""
import os
os.chdir('..')
import time
import json
import numpy as np
import pandas as pd
from Algoritmo_Genetico import AG
#%%
# Carga la informacion para entrenar el modelo
train = pd.read_csv("../data/train_z.csv").drop(columns=["TW", "BARRIO"])
validation = pd.read_csv("../data/validation_z.csv").drop(columns=["TW", "BARRIO"])
test = pd.read_csv("../data/test_z.csv").drop(columns=["TW", "BARRIO"])

# Divide conjunto de datos en entrenamiento y en prueba
X_train = train.drop(columns="Accidente")
Y_train = train["Accidente"]

X_val = validation.drop(columns="Accidente")
Y_val = validation["Accidente"]

X_test = test.drop(columns="Accidente")
Y_test = test["Accidente"]
#%%
count = 0
list_best_score = []
list_prob_mut = []
times = []
for prob_mut in np.arange(0.1,1,0.1):
    print('***'*20)
    print(f'Ejecuta escenario con probabilidad de mutacion {prob_mut}')
    time_ini = time.time()
    best_mod, best_score, bes_mod_entrenado, historia_score = AG(X_train,
                                                                  Y_train,
                                                                  X_val,
                                                                  Y_val,
                                                                  num_generacion=20,
                                                                  population_size=50,
                                                                  prob_seleccion=0.5,
                                                                  prob_mutacion=prob_mut,
                                                                  escenario=3,
                                                                  iteracion = count,
                                                                  desc = 'probMutacion')
    list_best_score.append(best_score)
    list_prob_mut.append(prob_mut)
    times.append((time.time()-time_ini)/60)
    count = count + 1
    
result = {'num_iteraciones': count-1,
          'prob_mutacion': list_prob_mut,
          'best_Scores': list_best_score,
          'tiempo': times}

with open(f'../data/output/resultado_escenario3_Mutacion.json', 'w') as json_file:
    json.dump(result, json_file)
