# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:07:35 2020

@author: pasal
"""
import os
os.chdir('..')
print(os.getcwd())
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
### Ejecucion del AG incluyendo busqueda exahustiva para el conjunto de 
### parametros seleccionado en el trabajo 1

print('***'*20)
print('INICIA EJECUCION AG INCLUYENDO BUSQUEDA EXAHUSTIVA')
time_ini = time.time()
best_mod, best_score, bes_mod_entrenado, historia_score = AG(X_train,
                                                              Y_train,
                                                              X_val,
                                                              Y_val,
                                                              num_generacion=20,
                                                              population_size=50,
                                                              prob_seleccion=0.5,
                                                              prob_mutacion=0.3,
                                                              escenario=1,
                                                              iteracion = 0,
                                                              desc = 'busquedaExahustiva',
                                                              BH = True,
                                                              bin_cont = False)

time_f = (time.time()-time_ini)/60

    
result = {'mod': best_mod,
          'best_Scores': best_score,
          'tiempo': time_f,
          'historia': historia_score}


with open(f'../data/output/resultado_BH.json', 'w') as json_file:
    json.dump(result, json_file)