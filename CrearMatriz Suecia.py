# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 12:40:05 2018

@author: Milton
"""
#%%
import pandas as pd
import numpy as np
#cuando hacemos cosas del estilo df.questionId[df.questionId == 19] = 230
#a pandas le da miedo de que la estes cagando y tira unos warning para decir
#'che fijate que podes estar sobreescribiendo cosas que no queres sobreescribir'
#con la siguiente linea se calla el warning y listo
pd.options.mode.chained_assignment = None

#frames tiene dos lugares, en el primero va a tener el df asociado a el archivo ww1
#y en el segundo va a tener el df asociado al ww2
frames = []
#Cargo los excels en una nueva matriz
for i in range(2): #Dentro de Range hay que poner el número de matrices que tengo, es decir la cantidad de .xlsx que tengo.
    #Cargo los excels en una estructura de pandas
    df = pd.read_excel(io='ww'+ str(i+1) +'.xlsx', sheet_name='export (ww'+ str(i+1) + ')')
    
    if i==0: #esta condición es para evitar problemas con el concatenate
        frames.append(df)
        
    else:
        #los user_id de ww2 arrancan desde ponele 0 entonces se superpondrian con los de
        #los user_id de ww1, para esto a toda la columna de user_id le sumamos
        #el ultimo de los user_id de ww1, asi evitamos tener 2 user_id iguales.
        df['user_id'] += np.max(frames[0]['user_id']) #Cambia los usrID de la pregunta que uso
        #en ww1 se preguntan primero A1 y A2 y por ultimo A3 y A4(lo mismo para B)
        #en ww2 se preguntan primero A3 y A4 y por ultimo A1 y A2(lo mismo para B)
        #entonces distinguimos esto con los forks
        #agregamos 2 a los forks de ww2 para que las preguntas de ww1 sean
        #fork 9 y 10 y las de ww2 sean 11 y 12.
        df['fork'] += 2
        """ Necesito que dejemos bien claro esto que no me esta saliendo redactar y no quiero mentir(lucas)
        En ww1 las qId asociadas a A1, A2, B1 y B2 son 19, 20, 21 y 22
        CREO QUE en ww2 las qId asociadas a A1, A2, B1 y B2 son 23, 39, 40 y 41
        ...
        """

        #los que en ww2 tienen qId 19,20,21,22 los cambio por uno arbitrario
        #que finalmente va a ser 23,39,40,41
        df.questionId[df.questionId == 19] = 230
        df.questionId[df.questionId == 20] = 390
        df.questionId[df.questionId == 21] = 400
        df.questionId[df.questionId == 22] = 410
        #los que en ww2 tienen qId 23,39,40,41 los cambio por 19,20,21,22
        #ya que en realidad estas son preguntas A1,A2,B1,B2 pero preguntadas
        #al final del cuestionario
        df.questionId[df.questionId == 23] = 19
        df.questionId[df.questionId == 39] = 20
        df.questionId[df.questionId == 40] = 21
        df.questionId[df.questionId == 41] = 22
        #las que eran en ww2 qId 19,20,21,22 ahora son 23,39,40,41
        #ya que en realidad era A3,A4,B3,B4 pero preguntadas al principio
        df.questionId[df.questionId == 230] = 23
        df.questionId[df.questionId == 390] = 39
        df.questionId[df.questionId == 400] = 40
        df.questionId[df.questionId == 410] = 41
        #appendeamos este df al lado del de ww1

        frames.append(df)
    print(i)
#df_N tiene ahora un df que es la "parte de arriba"(las primeras filas) el df de ww1
#y la "parte de abajo"(las ultimas filas) el df de ww2
df_N = pd.concat(frames, ignore_index=True) #ignore_index=True hace que se peguen bien las matrices

"""
sin ignore_index=true se estaban indexando mal 
los dataframe, conservaban sus indices y
eso hacia que al llamar un indice como por ejemplo en isinstance
se pida algo doble que tiene formato de pd.algunacosa y no
un string como estaba antes
"""
#df_Base = pd.read_excel('BaseSW.xlsx')
#%
#Esto estaba por si no funcaba la linea siguiente. Por ahora lo dejo
#ans = df_N.ValorRespondido.values
#A = np.where(df_N.questionId == 31)[0].astype(int)
#
#for j in A:
#    if ans[j] > 3:
#        ans[j] -= 1
#
#df_N.ValorRespondido = ans

"""    
df_N.ValorRespondido nos muestra la columna asociada a esta variable.

df_N.ValorRespondido[CONDICION] nos muestra los valores de la columna asociada
que cumplen con la condicion que se le pida.

df_N.ValorRespondido[CONDICION] -= 1 agarra los valores que nos muestra y les resta 1.

PREGUNTA: POR QUE HACEMOS ESTO?
la pregunta 31 es "¿que tan seguro es que votes por ese partido?" y le decimos que si
esa es la pregunta y la persona respondio algo mayor a 3, hay que restarle 1 a esa 
respuesta
"""
df_N.ValorRespondido[(df_N.questionId == 31) & (df_N.ValorRespondido > 3)] -=  1
#isistance se fija que el primer argumento sea de la clase o tipo del segundo argumento
#esto es un CONTROL de que los formatos esten bien
df_N = df_N[[isinstance(x,int) for x in df_N.user_id]]
df_N = df_N[[isinstance(x,int) for x in df_N.id]]
df_N = df_N[[isinstance(x,unicode) for x in df_N.Pais]]
#le saco el .tslib porque tira un warning de que es obsoleto y va a ser removido
#en futuras versiones PONER EN COMENTARIOS Y SACAR DE ACA
df_N = df_N[[isinstance(x,pd.Timestamp) for x in df_N.creado_el]]
del x
df_N.sort_values(by = 'user_id', kind = 'mergesort') #me parece que esto se cumple solo
#%%
FechaI = pd.Timestamp(2018,9,5,12,30,0) #fecha de inicio y finalizacion del experimento
#FechaF = dt.datetime(2018,6,17,8,0,0)

""" FILTRO
df_N = df_N [CONDICION] nos elimina las filas que NO cumplan con(nos deja las filas que SI cumplan con):
    df_N['OmitirDatos?'] != 1 si esto es 1 se deben omitir esos datos y hay que tirarlo
    df_N.EsUsuarioDePrueba != 1 si esto es 1 el usuario es de prueba y hay que tirarlo
    df_N.creado_el > FechaI si la respuesta es anterior a esta fecha hay que tirarlo
    df_N.Pais == 'Sweden' si el pais no es suecia hay que tirarlo
    
luego de este filtro nos queda el data frame sin datos que debieran ser omitidos por motivos
preestablecidos, usuarios de prueba, viejos o de otros paises.
"""
df_N = df_N[(df_N['OmitirDatos?'] != 1) & (df_N.EsUsuarioDePrueba != 1) & (df_N.creado_el > FechaI) & (df_N.Pais == 'Sweden')] 
CL = [] #CREO QUE Nivel de confianza promedio que presento un dado usuario
Pol = [] #CREO QUE Que tan polarizado esta el usuario

""" FILTRO
df_N.user_id.unique() es un array que se queda con los valores de
la columna user_id pero elimina los repetidos,
printeando df_N.user_id y df_N.user_id.unique() se entiende mejor.

CREO QUE lo primero es revisar que los usuarios no hayan contestado
dos veces la misma pregunta, esto se revisa en las lineas:
-para las preguntas
df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 0)] = df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 0)].drop_duplicates('questionId')
-para las repreguntas
df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 1)] = df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 1)].drop_duplicates('questionId')
donde se realiza esta distincion porque sino ya existirian repreguntas que, 
en principio, no queremos eliminar

PREGUNTA: df_N = df_N[~df_N.user_id.isna()] elimina la fila que tiene nan en user_id
no seria mejor eliminar al usuario directamente?
AH, ME PARECE que en
if df_N[df_N.user_id == i].shape != (24,16):
    df_N = df_N[df_N.user_id != i]
se hace eso, ya que las que se les borro, por ej, 1 fila repetida tienen shape (23,16)
"""

for i in df_N.user_id.unique():
    if np.mod(i,100)==0:
        print(i) #son 2500 aprox
    df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 0)] = df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 0)].drop_duplicates('questionId') #Elimino las preguntas duplicadas dejando solo la primera, cuando se eliminan quedan como Na
    df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 1)] = df_N[(df_N.user_id == i) & (df_N.Es_Repregunta == 1)].drop_duplicates('questionId') #Idem para las repreguntas
    df_N = df_N[~df_N.user_id.isna()] #Elimino las filas que habían quedado como Na
                    
    if df_N[df_N.user_id == i].shape != (24,16):
        df_N = df_N[df_N.user_id != i]
    
    else:
        CL.append((np.mean(df_N.ValorRespondido[(df_N.Es_Repregunta == 0) & (df_N.TipoDePregunta == 1) & (df_N.user_id == i)]) - np.mean(df_N.ValorRespondido[(df_N.Es_Repregunta == 0) & (df_N.TipoDePregunta == 2) & (df_N.user_id == i)]) + 100)/2)
        #Pol.append(np.mean(2*abs(df_N.ValorRespondido[(df_N.Es_Repregunta == 0) & (df_N.user_id == i)])))
        #no estoy seguro de que es pol pero no creo que tenga que tener los tipos de pregunta 0 para promediar con nada
        #tampoco se para que esta el 2*abs
        Pol.append(np.mean(df_N.ValorRespondido[(df_N.Es_Repregunta == 0) & (df_N.user_id == i) & (df_N.TipoDePregunta != 0)]))

df_users = pd.DataFrame({"CL": CL , "Pol":Pol})
df_users.index = df_N.user_id.unique()
#%%
"""
Bueno parece haber una necesidad de decirle al amigo df_N que tire todas las repreguntas,
esto charlemoslo de todas maneras
"""
df_N = df_N[df_N.Es_Repregunta == 0]
#%%

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
df_N.ValorRespondido[df_N.questionId == 19] = -df_N.ValorRespondido[df_N.questionId == 19] + 100
df_N.ValorRespondido[df_N.questionId == 23] = -df_N.ValorRespondido[df_N.questionId == 23] + 100

#Guarda la matriz lista para el análisis, y un vector Us que es una lista de los id de usuarios que existen
#DECISIÓN IMPORTANTE: Dejo solo las primeras 4 repreguntas de cada usuario
#Cambie los nans que eran ceros por ceros

#df_Base.to_pickle('BaseSW', protocol = 2)
df_N.to_pickle('MatrizTotal', protocol = 2)
df_users.to_pickle('MatrizUsuarios', protocol = 2)   
