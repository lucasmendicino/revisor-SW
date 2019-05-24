# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:25:27 2018
@author: Milt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy.stats import norm, linregress as lr, chi2
#%% Código para mostrarle a los Suecos en Suecia
#Las cosas importantes son: mean(A1,A2), mean(B1,B2), mean(A3, A4), mean(B3,B4), DR, Fork, usrID,   
#Scatter mean(1,2) vs mean(3,4) uno para los manipulados y uno para los no manipulados, linealizar y ver si la pendiendte es igual o distinta

#%%                     ESTO SE CORRE SI SE QUIERE USAR LA MATRIZ NUEVA
#%% Carga la matriz(la nueva hecha con pandas)
Datosdf = np.load('MatrizTotal', allow_pickle=True)
#reordeno las columnas del data frame para que a la hora de mirar columnas
#tengan los mismo numeritos que lo que veniamos usando con la otra matriz
Datosdf = Datosdf[['Pais','user_id','timer','fork','questionId',
                   'TipoDePregunta','ValorRespondido','Es_Repregunta',
                   'ValorRepregunta','ValorPresentadoPregunta',
                   'creado_el','confianza']]
#N es ahora un array de arrays tal y como lo era N en el caso anterior
N=Datosdf.values
"""
Recorremos la columna 4 que es columna del datasheet que tiene
el question id de las preguntas realizadas.
Pedimos el valor de la columna 6 que es la que tiene el valor
de la respuesta.
"""
A1 = N[np.where(N[:,4] == 19)[0].astype(int),6]
A2 = N[np.where(N[:,4] == 20)[0].astype(int),6] 
A3 = N[np.where(N[:,4] == 23)[0].astype(int),6]
#A4 = N[np.where(N[:,4] == 39)[0].astype(int),6] 
B1 = N[np.where(N[:,4] == 21)[0].astype(int),6] 
#B2 = N[np.where(N[:,4] == 22)[0].astype(int),6] 
B3 = N[np.where(N[:,4] == 40)[0].astype(int),6] 
B4 = N[np.where(N[:,4] == 41)[0].astype(int),6]
"""
Recorremos la columna 4 que es columna del datasheet que tiene
el question id de las preguntas realizadas.
Pedimos el valor de la columna 11 que es la que tiene el valor
de la confianza en la respuesta.
"""
ConfA1 = N[np.where(N[:,4] == 19)[0].astype(int),11]
ConfA2 = N[np.where(N[:,4] == 20)[0].astype(int),11] 
ConfA3 = N[np.where(N[:,4] == 23)[0].astype(int),11] 
ConfA4 = N[np.where(N[:,4] == 39)[0].astype(int),11] 
ConfB1 = N[np.where(N[:,4] == 21)[0].astype(int),11] 
ConfB2 = N[np.where(N[:,4] == 22)[0].astype(int),11] 
ConfB3 = N[np.where(N[:,4] == 40)[0].astype(int),11] 
ConfB4 = N[np.where(N[:,4] == 41)[0].astype(int),11] 
#%%
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

for i in range(len(A1)):
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
#%%                         FUNCIONES PARA SCATTER CON AJUSTE, CHI2 Y TEST Z
###                             REGRESION LINEAL CON GRAFICOS
def RegLineal(A,B,C,D,stringAB,stringCD,title,xlab,ylab):
    plt.figure(figsize=(5,4))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    xx=np.linspace(0,100,500)
    #scatter de respuestas iniciales vs finales cuando se manipula la respuesta inicial
    plt.scatter(A,B, label=stringAB, s=2, color='red', alpha=0.8)
    #regresion lineal utilizando cuadrados minimos para hallar los parametros de la recta
    slopeAB, interceptAB, r_value, p_value, std_errAB=lr(A,B)
    plt.plot(xx,slopeAB*xx+interceptAB, color='red', alpha=0.8)
    #scatter de respuestas iniciales vs finales cuando NO se manipula la respuesta inicial
    plt.scatter(C,D, label=stringCD, s=2, color='green', alpha=0.8)
    #regresion lineal utilizando cuadrados minimos para hallar los parametros de la recta
    slopeCD, interceptCD, r_valueNM, p_valueNM, std_errCD=lr(C,D)
    plt.plot(xx,slopeCD*xx+interceptCD, color='green', alpha=0.8)
    plt.legend(loc='upper left')
    plt.tight_layout()
    return slopeAB, interceptAB, std_errAB, slopeCD, interceptCD, std_errCD
###                         TEST CHICUADRADO CON GRAFICO
def ChiCuadrado(A,B,C,D,stringAB,stringCD, slopeAB, interceptAB, slopeCD, interceptCD, title):
    plt.figure(figsize=(5,4))
    plt.title(title)
    yAB = slopeAB * A + interceptAB
    TAB = np.sum((B - yAB)**2/yAB) #estadistico
    yCD= slopeCD * C + interceptCD
    TCD= np.sum((D - yCD)**2/yCD) #estadistico
    df=len(B)-1
    def chi2Acumulada(X):
        return chi2.cdf(X,df)
    x = np.linspace(df-50*np.sqrt(df),df+50*np.sqrt(df),100)
    plt.plot(x, chi2.pdf(x, df),
       'r-', lw=2, alpha=0.6)
    plt.vlines(TAB,0,max(chi2.pdf(x, df)), color='red', label = stringAB + ' %d' %TAB)
    plt.vlines(TCD,0,max(chi2.pdf(x, df)),color='green', label = stringCD + ' %d' %TCD)
    pvAB=1-chi2.cdf(TAB,df) #pvalor para AB
    pvCD=1-chi2.cdf(TCD,df) #pvalor para CD
    plt.legend(loc='upper left')
    plt.tight_layout()
    return pvAB, TAB, pvCD, TCD
###                                 TEST Z
def testZ(slopeAB, std_errAB, slopeCD, std_errCD, title):
    plt.figure(figsize=(5,4))
    plt.title(title)
    sigma=np.sqrt(std_errAB**2+std_errCD**2)
    x = np.linspace(-10*sigma, 10*sigma, 1000)
    #defino estas dos para que sea menos feo abajo
    def gauss(X):
        g = norm.pdf(X,0,np.sqrt(std_errAB**2+std_errCD**2))
        return g
    def gaussAcumulada(X):
        gAc = norm.cdf(X,0,np.sqrt(std_errAB**2+std_errCD**2))
        return gAc
    #dibujo la distribucion del estadistico
    plt.plot(x, gauss(x),
           'r-', lw=2, alpha=0.6, label='gaussiana')
    U=(slopeCD-slopeAB)/np.sqrt(std_errAB**2+std_errCD**2) #estadistico
    plt.vlines(U,0,max(gauss(x)), color='blue') #muestro el estadistico en el dibujo
    if U > 0:
        pv=1-(2*(gaussAcumulada(U)-gaussAcumulada(0))) #pvalor para U
        #dibujo el area que va a ser la mitad del p-valor por ser a dos colas
        xx=np.linspace(U,x[-1],15)
        plt.fill_between(xx, 0, gauss(xx),color='r', alpha=0.4)
    else:
        pv=1-(2*(gaussAcumulada(0)-gaussAcumulada(U))) #pvalor para U
        #dibujo el area que va a ser la mitad del p-valor por ser a dos colas
        xx=np.linspace(x[1],U,15)
        plt.fill_between(xx, 0, gauss(xx),color='r', alpha=0.4)
    patch_pv = mpatches.Patch(color='blue', label=r'el pvalor es %f' %(pv), alpha=0.4)
    patch_area = mpatches.Patch(color='red', label=r'el area bajo la curva es %f' %(pv/2), alpha=0.4)
    plt.legend(handles=[patch_pv, patch_area], loc='upper left')
    plt.tight_layout()
    return pv, U
#%%                             All topics
#regresion lineal
slope, intercept, std_err, slopeNM, interceptNM, std_errNM = RegLineal(Ti, Tf, TNMi, TNMf, 
                'manipulados','no manipulados','All topics','Opening questions agreement',
                'Final questions agreement')
#testChiCuadrado
pvTiTf, TTiTf, pvTNMiTNMf, TTNMiTNMf = ChiCuadrado(Ti,Tf,TNMi,TNMf,
                 'manipulados','no manipulados', slope, intercept, 
                 slopeNM, interceptNM, 'All Topics')
#testZ
pV, u = testZ(slope, std_err, slopeNM, std_errNM,'All topics')
#%%                             Inmigration
AiM=[] #las preguntas del tema A iniciales cuando se manipula
AfM=[] #las preguntas del tema A finales cuando se manipula
ANMi=[] #las preguntas del tema A iniciales cuando NO se manipula
ANMf=[] #las preguntas del tema A finales cuando NO se manipula
for i in range(len(Ai)):
    #Estos forks manipulan las respuestas del tema A
    if F[i] == 9 or F[i]==11:
        AiM.append(Ai[i])
        AfM.append(Af[i])
    #Estos no
    else:
        ANMi.append(Ai[i])
        ANMf.append(Af[i])
Ai=np.array(Ai)
Af=np.array(Af)
ANMi=np.array(ANMi)
ANMf=np.array(ANMf)
#regresion lineal
slope, intercept, std_err, slopeNM, interceptNM, std_errNM = RegLineal(Ai,Af,ANMi,ANMf,
          'manipulados','no manipulados','Inmigration',
          'Opening questions agreement','Final questions agreement')
#testChiCuadrado
pvAiAf, TAiAf, pvANMiANMf, TANMiANMf = ChiCuadrado(Ai,Af,ANMi,ANMf,
                 'manipulados','no manipulados', slope, intercept,
                 slopeNM, interceptNM, 'Inmigration')
#TestZ
pV, u = testZ(slope, std_err, slopeNM, std_errNM,'Inmigration')
#%%                              Ecology
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
#regresion lineal
slope, intercept, std_err, slopeNM, interceptNM, std_errNM = RegLineal(Bi,Bf,BNMi,BNMf,
          'manipulados','no manipulados','Ecology',
          'Opening questions agreement','Final questions agreement')
#testChiCuadrado
pvTiTf, TTiTf, pvTNMiTNMf, TTNMiTNMf = ChiCuadrado(Bi,Bf,BNMi,BNMf,
                 'manipulados','no manipulados', slope, intercept,
                 slopeNM, interceptNM, 'Ecology')
#TestZ
pV, u = testZ(slope, std_err, slopeNM, std_errNM,'Ecology')

#%%                                     TEST DE RUNS
#%%
##Aviso!!! Se debe correr este codigo con la matriz sin las repreguntas 
#(en CrearMatriz se tiene esa opcion, sino se debe filtrar por repregunta)

#calculo el estadistico de runs para dos conjuntos A y B
def EstRuns(A,B):
    #los redefino para etiquetar cada elemento con el conjunto del cual provienen
        #A es el conjunto asociado al 0
    labelA=np.reshape(A,(len(A),1))
    labelA=np.insert(labelA,1,0,axis=1)
        #B es el conjunto asociado al 1
    labelB=np.reshape(B,(len(B),1))
    labelB=np.insert(labelB,1,1,axis=1)
    #creo un conjunto con ambos para sortearlo
    conjunto=np.append(labelA,labelB,axis=0)
    conjunto=conjunto[conjunto[:,0].argsort()]
    #cuento la cantidad de runs (cambios de conjunto de A a B y viceversa)
    runs=0
    for i in range(len(conjunto)):
        if i==0 or conjunto[i][1]==conjunto[i-1][1]:
            pass
        else:
            runs+=1
    return runs

#defino el calculo del p-valor con una aproximacion gaussiana (aproxima la distribucion del estadistico como una normal
#de esperanza y varianza dadas por las formulas del estadistico). La funcion tambien plotea, tomando un título
def PvalG(A,B,runs,title=None):
    N=len(A)
    M=len(B) 
    #calculamos la esperanza y la varianza de la distribucion (usando las formulas del test runs)
    mu=2*N*M/(N+M)+1
    sigma=(2*N*M*(2*N*M-N-M)/((N+M)**2*(N+M-1)))**(0.5)
    #obtenemos la gaussiana y la ploteamos
    dom=np.linspace(mu-30*sigma,mu+30*sigma,10000)
    gauss=norm.pdf(dom,mu,sigma)
    #calculamos el p-valor
    pValue=0
    i=0
    while dom[i]<=runs:
        pValue+=gauss[i]
        i+=1
    norma=pValue
    for j in range(i+1,len(dom),1):
        norma+=gauss[j]
    plt.plot(dom, gauss, label='Distribución Normal')
    plt.axvline(x=runs,linestyle='--', color='r', label='Estadístico de Runs: '+str(runs))
    pValue/=norma
    plt.plot([], [], ' ', label='P-valor: '+str(np.format_float_scientific(pValue, precision=2)))
    plt.xlabel('Estadísticos de Runs')
    plt.ylabel('Probabilidad')
    plt.title(title)
    plt.legend()
    plt.savefig(title+'.png', dpi=150)
    plt.show()
    return pValue

#guarda las respuestas iniciales y finales (i y f respectivamente) de los usuarios manipulados que hayan contestado 
#con un promedio de agreement mayor a 50 EN SU PRIMERA RESPUESTA 
MMi=np.array([])
MMf=np.array([])
#guarda las respuestas iniciales y finales (i y f respectivamente) de los usuarios no manipulados que hayan contestado 
#con un promedio de agreement mayor a 50 EN SU PRIMERA RESPUESTA 
NMMi=np.array([])
NMMf=np.array([])


#guarda las respuestas iniciales y finales (i y f respectivamente) de los usuarios manipulados que hayan contestado 
#con un promedio de agreement menor a 50 EN SU PRIMERA RESPUESTA 
Mmi=np.array([])
Mmf=np.array([])
#guarda las respuestas iniciales y finales (i y f respectivamente) de los usuarios no manipulados que hayan contestado 
#con un promedio de agreement menor a 50 EN SU PRIMERA RESPUESTA 
NMmi=np.array([])
NMmf=np.array([])

#recorro todas las respuestas manipuladas de todos los usuarios

##PREGUNTAR: lo hace hasta 1981 porque despues de eso me tira error de index (revisar, porque el total es 3024)
for i in range(len(A1)):
    #guardo aquellos cuyo agreement inicial haya sido mayor o igual a 50 (manipulados)
    if Ti[i]>=50:
        MMi = np.append(MMi, Ti[i])
        MMf = np.append(MMf, Tf[i])
    #guardo aquellos cuyo agreement inicial haya sido menor a 50
    else:
        Mmi = np.append(Mmi, Ti[i])
        Mmf = np.append(Mmf, Tf[i])
    #guardo aquellos cuyo agreement inicial haya sido mayor o igual a 50 (no manipulados)
    if TNMi[i]>=50:
        NMMi = np.append(MMi, TNMi[i])
        NMMf = np.append(MMf, TNMf[i])
    #guardo aquellos cuyo agreement inicial haya sido menor a 50
    else:
        NMmi = np.append(Mmi, TNMi[i])
        NMmf = np.append(Mmf, TNMf[i])
 
    

#estadistico y p-valor del grupo de mayores de 50
runsM=EstRuns(MMi,MMf)
pValM=PvalG(MMi,MMf,runsM,'Manipulados con agreement inicial mayor a 50')

#estadistico y p-valor del grupo de menores de 50
runsm=EstRuns(Mmi,Mmf)
pValm=PvalG(Mmi,Mmf,runsm,'Manipulados con agreement inicial menor a 50')

#estadistico y p-valor del grupo de mayores de 50
runsNM=EstRuns(NMMi,NMMf)
pValNM=PvalG(NMMi,NMMf,runsNM,'No manipulados con agreement inicial mayor a 50')

#estadistico y p-valor del grupo de menores de 50
runsNm=EstRuns(NMmi,NMmf)
pValNm=PvalG(NMmi,NMmf,runsNm,'No manipulados con agreement inicial menor a 50')



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
