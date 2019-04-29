# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:25:27 2018

@author: Milt
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%% Carga la matriz

Datos = np.load('MatrizLimpiaConCeros.npz')
N = Datos['N'] 
Us= Datos['Us']

#%% Código para mostrarle a los Suecos en Suecia
#Las cosas importantes son: mean(A1,A2), mean(B1,B2), mean(A3, A4), mean(B3,B4), DR, Fork, usrID,   
#Scatter mean(1,2) vs mean(3,4) uno para los manipulados y uno para los no manipulados, linealizar y ver si la pendiendte es igual o distinta
"""
Recorremos la columna 4 que es columna del datasheet que tiene
el question id de las preguntas realizadas.
Pedimos el valor de la columna 6 que es la que tiene el valor
de la respuesta.
"""
A1 = - N[np.where(N[:,4] == 19)[0].astype(int),6] + 100
A1 = A1[np.arange(0,len(A1),2)]
A2 = N[np.where(N[:,4] == 20)[0].astype(int),6] 
A2 = A2[np.arange(0,len(A2),2)]
A3 = - N[np.where(N[:,4] == 23)[0].astype(int),6] + 100
#A4 = N[np.where(N[:,4] == 39)[0].astype(int),6] 
B1 = N[np.where(N[:,4] == 21)[0].astype(int),6] 
B1 = B1[np.arange(0,len(B1),2)]
#B2 = N[np.where(N[:,4] == 22)[0].astype(int),6] 
#B2 = B2[np.arange(0,len(B2),2)]
B3 = N[np.where(N[:,4] == 40)[0].astype(int),6] 
B4 = N[np.where(N[:,4] == 41)[0].astype(int),6] 
"""
Recorremos la columna 4 que es columna del datasheet que tiene
el question id de las preguntas realizadas.
Pedimos el valor de la columna 11 que es la que tiene el valor
de la confianza en la respuesta.
"""
ConfA1 = N[np.where(N[:,4] == 19)[0].astype(int),11]
ConfA1 = ConfA1[np.arange(0,len(ConfA1),2)]
ConfA2 = N[np.where(N[:,4] == 20)[0].astype(int),11] 
ConfA2 = ConfA2[np.arange(0,len(ConfA2),2)]
ConfA3 = N[np.where(N[:,4] == 23)[0].astype(int),11] 
ConfA4 = N[np.where(N[:,4] == 39)[0].astype(int),11] 
ConfB1 = N[np.where(N[:,4] == 21)[0].astype(int),11] 
ConfB1 = ConfB1[np.arange(0,len(ConfB1),2)]
ConfB2 = N[np.where(N[:,4] == 22)[0].astype(int),11] 
ConfB2 = ConfB2[np.arange(0,len(ConfB2),2)]
ConfB3 = N[np.where(N[:,4] == 40)[0].astype(int),11] 
ConfB4 = N[np.where(N[:,4] == 41)[0].astype(int),11] 
"""
Recorremos la columna 4 que es columna del datasheet que tiene
el question id de las preguntas realizadas.
Pedimos el valor de fork en el que fue realizada cada pregunta
"""
F = N[np.where(N[:,4] == 62)[0].astype(int),3] 

#promedio entre las dos respuestas iniciales del tema A
Ai = np.array([])
#promedio entre las dos respuestas finales del tema A
Af = np.array([])
#promedio entre las dos respuestas iniciales del tema B
Bi = np.array([])
#promedio entre las dos respuestas finales del tema B
Bf = np.array([])
"""
Ti y Tf guardan el promedio de las respuestas Ai y Bi cuando se manipula
las respuestas a Ai(Fork 9 y 11) o Bi(Fork 10 y 12) respectivamente.
TNMi y TNMf guardan el promedio de las respuestas Ai y Bi cuando NO se manipula
las respuestas a Ai(Fork 10 y 12) o Bi(Fork 9 y 11) respectivamente.
"""
Ti = np.array([])
Tf = np.array([])
TNMi = np.array([])
TNMf = np.array([])
ConfA = np.array([])
ConfB = np.array([])

for i in range(len(Us)):
    #en estos dos fork (A1, A2) son iniciales y (A3, A4) son finales
    #en estos dos fork (B1, B2) son iniciales y (B3, B4) son finales
    if F[i]==9 or F[i]==10:
        Ai = np.append(Ai, np.mean((A1[i],A2[i])))
        Bi = np.append(Bi, B1[i])#np.mean((B1[i],B2[i])))
        Af = np.append(Af, A3[i])#np.mean((A4[i],A3[i])))
        Bf = np.append(Bf, np.mean((B3[i],B4[i])))
    #en estos dos fork (A3, A4) son iniciales y (A1, A2) son finales
    #en estos dos fork (B3, B4) son iniciales y (B1, B2) son finales 
    else:
        Ai = np.append(Ai, A3[i])#np.mean((A4[i],A3[i])))
        Bi = np.append(Bi, np.mean((B3[i],B4[i])))
        Af = np.append(Af, np.mean((A1[i],A2[i])))
        Bf = np.append(Bf, B1[i])#np.mean((B1[i],B2[i])))

    if F[i] == 9 or F[i] == 11:
        Ti = np.append(Ti,Ai[i])
        Tf = np.append(Tf,Af[i])
        TNMi = np.append(TNMi,Bi[i])
        TNMf = np.append(TNMf,Bf[i])
    else:
        Ti = np.append(Ti,Bi[i])
        Tf = np.append(Tf,Bf[i])
        TNMi = np.append(TNMi,Ai[i])
        TNMf = np.append(TNMf,Af[i])   

#Cada componente de M es una repregunta distinta y para cada repregunta pongo un 0 si estuvo manipulada y un 1 si no
#Cada componente de D es una repregunta distinta y para cada repregunta pongo un 0 si la detecté y un 1 si no

M = np.array([])
D = np.array([])
"""
Recorremos la columna 7 que es columna del datasheet que indica
si la pregunta realizada es una repregunta con un 1, si no es
repregunta hay un 0.
Guardamos en A los indices para los cuales el valor es 1.
"""
A = np.where(N[:,7]==1)[0].astype(int) #Tomo los indices de las repreguntas

#miramos solamente las filas que sean repregunta
for i in A:
    #si manipulamos a la persona
    #es decir que le presentamos en una repregunta
    #un valor (N[i,9]) distinto al que contesto al principio (N[i,6])
    if N[i,6] != N[i,9]:
        #guardamos un 1 en M
        M = np.append(M, 1)
        #la columna 8 contiene -1 si la persona en una repregunta no cambia el valor presentado
        #y tiene el valor de respuesta si decide modificar el valor presentado en la repregunta
        
        #si N[i,8] y N[i,9] estan de distinto lado del 50 es porque la persona nota algo extraño
        #y cambia el valor propuesto pero sin embargo mantiene una postura contraria a la de
        #la primera vez
        
        #si N[i,8] != -1 es porque decidio cambiar lo que se le presento
        
        #es decir que si se cumplen estas dos consideramos que detecto el engaño
        if (N[i,8]-50)*(N[i,9]-50) < 0 and N[i,8] != -1:
            #guardamos un 1 en D
            D = np.append(D,1)
        #si N[i,8] y N[i,9] estan del mismo lado del 50 es porque la persona no nota el engaño
        #y no devuelve el cursor al "lado correcto" de la escala del 1 al 100
        
        #si N[i,8] == -1 es porque no detecto el engaño
        else:
            #guardamos un 0 en D
            D = np.append(D,0)
    #si no intentamos manipular a la persona
    #es decir que le presentamos en una repregunta
    #el mismo valor al que contesto al principio
    else:
        M = np.append(M,0)    

DR = np.sum(D)/np.sum(M) #Detection rate

#%%------------------------------------%%#
"""                                 RECORDEMOS
-Ti guarda el promedio de las respuestas iniciales Ai cuando el Fork es 9 y 11
y Bi cuando el Fork es 10 y 12, es decir cuando las respuestas iniciales 
de cada fork van a ser manipuladas.
Tf guarda la respuesta a las preguntas finales asociadas al mismo tema 
en estos casos.

-TNMi guarda el promedio de las respuestas inciales Ai cuando el Fork
es 10 y 12 y Bi cuando el Fork es 9 y 11, es decir cuando las respuestas
iniciales de cada fork NO van a ser manipuladas.
TNMf guarda la respuesta a las preguntas finales asociadas al mismo tema 
en estos casos.
"""

from scipy.stats import linregress as lr#importo aca para que se entienda que es lr
#despues lo mandamos para arriba en caso de querer usarlo

Tfin = np.zeros(10)
k = np.zeros(10) # esto es para normalizar

TNMfin = np.zeros(10)
kNM = np.zeros(10) # esto es para normalizar

for i in range(len(Ti)):
    #estoy bastante seguro de que esto guarda las cosas manipuladas(lucas)
    Tfin[int(Ti[i]/11)] = Tfin[int(Ti[i]/11)] + Tf[i]
    k[int(Ti[i]/11)] = k[int(Ti[i]/11)] + 1
    #estoy bastante seguro de que esto guarda las cosas NO manipuladas(lucas)
    TNMfin[int(TNMi[i]/11)] = TNMfin[int(TNMi[i]/11)] + TNMf[i]
    kNM[int(TNMi[i]/11)] = kNM[int(TNMi[i]/11)] + 1 #creo que estoy contaba mal la cantidad
    #de datos que habia en cada intervalo

Tfin = Tfin/k
TNMfin = TNMfin/kNM #me parece que estaba mal normalizado esto antes(lucas)

plt.figure() #dibu
plt.xlabel('Opening questions agreement')
plt.ylabel('Final questions agreement')
plt.title('All topics')
#estoy bastante seguro de que los labels son al reves aca(lucas)
#ya que durante el resto del codigo se hizo referencia a las cosas
#no manipuladas con NM, me tiene confundido esto(lucas)
#plt.scatter(['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'],Tfin, label = 'Non manipulated')
#plt.scatter(['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'],TNMfin, label = 'Manipulated')
#scatter de promedios que hizo Milton cambiando los labels(lucas)
plt.scatter(np.arange(5,105,10),Tfin, label = 'Manipulated', color='blue', alpha=0.8)
plt.scatter(np.arange(5,105,10),TNMfin, label = 'Non manipulated', color='orange', alpha=0.8)
#regresion lineal utilizando cuadrados minimos para hallar los parametros de la recta
#usando como datos los promedios(no se ni para que lo hice, miterio)(lucas)
xx=np.linspace(0,100,500)
slope, intercept, r_value, p_value, std_err=lr(np.arange(5,105,10),Tfin)
plt.plot(xx,slope*xx+intercept, color='blue', alpha=0.8)
slopeNM, interceptNM, r_valueNM, p_valueNM, std_errNM=lr(np.arange(5,105,10),TNMfin)
plt.plot(xx,slopeNM*xx+interceptNM, color='orange', alpha=0.8)

#scatter de respuestas iniciales vs finales cuando se manipula la respuesta inicial
plt.scatter(Ti,Tf, label='Ti vs Tf(manipulados)', s=2, color='red', alpha=0.8)
#regresion lineal utilizando cuadrados minimos para hallar los parametros de la recta
slope, intercept, r_value, p_value, std_err=lr(Ti,Tf)
plt.plot(xx,slope*xx+intercept, color='red', alpha=0.8)
#scatter de respuestas iniciales vs finales cuando NO se manipula la respuesta inicial
plt.scatter(TNMi,TNMf, label='TNMi vs TNMf(no manipulados)', s=2, color='green', alpha=0.8)
#regresion lineal utilizando cuadrados minimos para hallar los parametros de la recta
slopeNM, interceptNM, r_valueNM, p_valueNM, std_errNM=lr(TNMi,TNMf)
plt.plot(xx,slopeNM*xx+interceptNM, color='green', alpha=0.8)

plt.legend()
#%%------------------------------------%%#
Afin = np.zeros(10)
k = np.zeros(10)

ANMfin = np.zeros(10)
kNM = np.zeros(10)

for i in range(len(Ai)):
    if F[i] == 9 or F[i]==11:
        Afin[int(Ai[i]/11)] = Afin[int(Ai[i]/11)] + Af[i]
        k[int(Ai[i]/11)] = k[int(Ai[i]/11)] + 1       
    else:
        ANMfin[int(Ai[i]/11)] = ANMfin[int(Ai[i]/11)] + Af[i]
        kNM[int(Ai[i]/11)] = kNM[int(Ai[i]/11)] + 1  

Afin = Afin/k
ANMfin = ANMfin/k

plt.figure()
plt.xlabel('Opening questions agreement')
plt.ylabel('Final questions agreement')
plt.title('Inmigration')
plt.scatter(['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'],Afin, label = 'Non manipulated')
plt.scatter(['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'],ANMfin, label = 'Manipulated')
plt.legend()
plt.savefig('An',dpi=200)
    
#%%------------------------------------%%#
    
Bfin = np.zeros(10)
k = np.zeros(10)

BNMfin = np.zeros(10)
kNM = np.zeros(10)

for i in range(len(Bi)):
    if F[i] == 10 or F[i]==12:
        Bfin[int(Bi[i]/11)] = Bfin[int(Bi[i]/11)] + Bf[i]
        k[int(Bi[i]/11)] = k[int(Bi[i]/11)] + 1       
    else:
        BNMfin[int(Bi[i]/11)] = BNMfin[int(Bi[i]/11)] + Bf[i]
        kNM[int(Bi[i]/11)] = kNM[int(Bi[i]/11)] + 1  

Bfin = Bfin/k
BNMfin = BNMfin/kNM

BiM=[] #las preguntas del tema B iniciales cuando se manipula
BfM=[] #las preguntas del tema B finales cuando se manipula
BNMi=[] #las preguntas del tema B iniciales cuando NO se manipula
BNMf=[] #las preguntas del tema B finales cuando NO se manipula
for i in range(len(Bi)):
    #Estos forks manipulan las respuestas del tema B
    if F[i] == 10 or F[i]==12:
        BiM.append(Bi[i])
        BfM.append(Bf[i])
    #Estos no
    else:
        BNMi.append(Bi[i])
        BNMf.append(Bf[i])

Bi=np.array(Bi)
Bf=np.array(Bf)
BNMi=np.array(BNMi)
BNMf=np.array(BNMf)

plt.figure()
plt.xlabel('Opening questions agreement')
plt.ylabel('Final questions agreement')
plt.title('Ecology')

#scatter de promedios que hizo Milton cambiando los labels(lucas)
plt.scatter(np.arange(5,105,10),Bfin, label = 'Manipulated', color='blue', alpha=0.8)
plt.scatter(np.arange(5,105,10),BNMfin, label = 'Non manipulated', color='orange', alpha=0.8)
#regresion lineal utilizando cuadrados minimos para hallar los parametros de la recta
#usando como datos los promedios(no se ni para que lo hice, miterio)(lucas)
xx=np.linspace(0,100,500)
slope, intercept, r_value, p_value, std_err=lr(np.arange(5,105,10),Bfin)
plt.plot(xx,slope*xx+intercept, color='blue', alpha=0.8)
slopeNM, interceptNM, r_valueNM, p_valueNM, std_errNM=lr(np.arange(5,105,10),BNMfin)
plt.plot(xx,slopeNM*xx+interceptNM, color='orange', alpha=0.8)

#scatter de respuestas iniciales vs finales cuando se manipula la respuesta inicial
plt.scatter(Bi,Bf, label='Bi vs Bf(manipulados)', s=2, color='red', alpha=0.8)
#regresion lineal utilizando cuadrados minimos para hallar los parametros de la recta
slope, intercept, r_value, p_value, std_err=lr(Bi,Bf)
plt.plot(xx,slope*xx+intercept, color='red', alpha=0.8)
#scatter de respuestas iniciales vs finales cuando NO se manipula la respuesta inicial
plt.scatter(BNMi,BNMf, label='BNMi vs BNMf(no manipulados)', s=2, color='green', alpha=0.8)
#regresion lineal utilizando cuadrados minimos para hallar los parametros de la recta
slopeNM, interceptNM, r_valueNM, p_valueNM, std_errNM=lr(BNMi,BNMf)
plt.plot(xx,slopeNM*xx+interceptNM, color='green', alpha=0.8)

plt.legend()
#%%------------------------------------%%#
Adif = np.zeros(10)
k = np.zeros(10)

ANMdif = np.zeros(10)
kNM = np.zeros(10)

for i in range(len(Ai)):
    if F[i] == 9 or F[i]==11:
        Adif[int(ConfA[i]/11)] = Adif[int(ConfA[i]/11)] + abs(Ai[i]-Af[i])
        k[int(ConfA[i]/11)] = k[int(ConfA[i]/11)] + 1       
    else:
        ANMdif[int(ConfA[i]/11)] = ANMdif[int(ConfA[i]/11)] + abs(Ai[i]-Af[i])
        kNM[int(ConfA[i]/11)] = kNM[int(ConfA[i]/11)] + 1  

Adif = Adif/k
ANMdif = ANMdif/k

plt.figure()
plt.xlabel('Confidence')
plt.ylabel('Variation of agreement')
plt.title('Confidence, inmigration')
plt.scatter(['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'],Adif, label = 'Non manipulated')
plt.scatter(['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'],ANMdif, label = 'Manipulated')
plt.legend()
plt.savefig('AConfn',dpi=200)
#%%------------------------------------%%#
Bdif = np.zeros(10)
k = np.zeros(10)

BNMdif = np.zeros(10)
kNM = np.zeros(10)

for i in range(len(Ai)):
    if F[i] == 10 or F[i]==12:
        Bdif[int(ConfB[i]/11)] = Bdif[int(ConfB[i]/11)] + abs(Bi[i]-Bf[i])
        k[int(ConfB[i]/11)] = k[int(ConfB[i]/11)] + 1       
    else:
        BNMdif[int(ConfB[i]/11)] = BNMdif[int(ConfB[i]/11)] + abs(Bi[i]-Bf[i])
        kNM[int(ConfB[i]/11)] = kNM[int(ConfB[i]/11)] + 1  

Bdif = Bdif/k
BNMdif = BNMdif/k

plt.figure()
plt.xlabel('Confidence')
plt.ylabel('Variation of agreement')
plt.title('Confidence, ecology')
plt.scatter(['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'],Bdif, label = 'Non manipulated')
plt.scatter(['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'],BNMdif, label = 'Manipulated')
plt.legend()
plt.savefig('BConfn',dpi=200)    

#%% Lo que está en el Word
#Initial Questions (Histogram of Initial Question by gender)
