# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 12:40:05 2018

@author: Milton
"""
#%%
import pandas as pd
import numpy as np

frames = []
#Cargo los excels en una nueva matriz
for i in range(2): #Dentro de Range hay que poner el número de matrices que tengo, 17 para chile, 1 para Colombia
    df = pd.read_excel(io='ww'+ str(i+1) +'.xlsx', sheet_name='export (ww'+ str(i+1) + ')')  #Cargo los excels en una estructura de pandas
    
    if i==0: #esta condición es para evitar problemas con el concatenate
        frames.append(df) #df.values es el array de numpy contenido en la estructura pandas. Este array tiene todo el excel menos los titulos 
        
    else:
        df['user_id'] += np.max(frames[0]['user_id']) #Cambia los usrID de la pregunta que uso
        df['fork'] += 2
        df.questionId[df.questionId == 19] = 23
        df.questionId[df.questionId == 20] = 41
        df.questionId[df.questionId == 21] = 40
        df.questionId[df.questionId == 22] = 41
        
        frames.append(df)
        df_N = pd.concat(frames)
    
    print(i) 

df_Base = pd.read_excel('BaseSW.xlsx')

#Esto estaba por si no funcaba la linea siguiente. Por ahora lo dejo
#ans = df_N.ValorRespondido.values
#A = np.where(df_N.questionId == 31)[0].astype(int)
#
#for j in A:
#    if ans[j] > 3:
#        ans[j] -= 1
#
#df_N.ValorRespondido = ans
    
df_N.ValorRespondido[(df_N.questionId == 31) & (df_N.ValorRespondido > 3)] -=  1    
    
df_N = df_N[[isinstance(x,int) for x in df_N.user_id]]
df_N = df_N[[isinstance(x,int) for x in df_N.id]]
df_N = df_N[[isinstance(x,str) for x in df_N.Pais]]
df_N = df_N[[isinstance(x,pd.tslib.Timestamp) for x in df_N.creado_el]]

df_N.sort_values(by = 'user_id', kind = 'mergesort')


FechaI = pd.tslib.Timestamp(2018,9,5,12,30,0) #fecha de inicio y finalizacion del experimento (para colombia del 13/6 al 17/6)
#FechaF = dt.datetime(2018,6,17,8,0,0)

df_N = df_N[(df_N['OmitirDatos?'] != 1) & (df_N.EsUsuarioDePrueba != 1) & (df_N.creado_el > FechaI) & (df_N.Pais == 'Sweden')] 

CL = [] #Vector donde indica las repreguntas que ya respondió el usuario
Pol = []

for i in df_N.user_id.unique():
    
    df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 0)] = df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 0)].drop_duplicates('questionId') #Elimino las preguntas duplicadas dejando solo la primera, cuando se eliminan quedan como Na
    df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 1)] = df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 1)].drop_duplicates('questionId') #Idem para las repreguntas
    df_N = df_N[~df_N.user_id.isna()] #Elimino las filas que habían quedado como Na
                    
    if df_N[df_N.user_id == i].shape != (24,16):
        df_N = df_N[df_N.user_id != i]
    
    else:
        CL.append((np.mean(df_N.ValorRespondido[(df_N.Es_Repregunta == 0) & (df_N.TipoDePregunta == 1) & (df_N.user_id == i)]) - np.mean(df_N.ValorRespondido[(df_N.Es_Repregunta == 0) & (df_N.TipoDePregunta == 2) & (df_N.user_id == i)]) + 100)/2)
        Pol.append(np.mean(2*abs(df_N.ValorRespondido[(df_N.Es_Repregunta == 0) & (df_N.user_id == i)])))
        
    
df_users = pd.DataFrame({"CL": CL , "Pol":Pol})
df_users.index = df_N.user_id.unique()


#Meto 0s donde deberían haber 0s y hay nulos
df_N.ValorRepregunta[df_N.ValorRepregunta.isna()] = 0
df_N.ValorPresentadoPregunta[df_N.ValorPresentadoPregunta.isna()] = 0
df_N.confianza[df_N.confianza.isna()] = 0
df_N.Es_Repregunta[df_N.Es_Repregunta.isna()] = 0

del df_N['id']
del df_N['comentarios']
del df_N['OmitirDatos?']
del df_N['EsUsuarioDePrueba']

#Revierto las preguntas que lo necesitan
df_N.questionId[df_N.questionId == 19] = -df_N.questionId[df_N.questionId == 19] + 100
df_N.questionId[df_N.questionId == 23] = -df_N.questionId[df_N.questionId == 23] + 100

#Guarda la matriz lista para el análisis, y un vector Us que es una lista de los id de usuarios que existen
#DECISIÓN IMPORTANTE: Dejo solo las primeras 4 repreguntas de cada usuario
#Cambie los nans que eran ceros por ceros

df_Base.to_pickle('BaseSW', protocol = 2)
df_N.to_pickle('MatrizTotal', protocol = 2)
df_users.to_pickle('MatrizUsuarios', protocol = 2)   

                
    