import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm, chi2, linregress as lr

#%%                                     TEST DE RUNS
"""Este test se usa para averiguar si dos conjuntos provienen de distribuciones con la misma esperanza. Consiste en 
tomar ambos grupos, ordenarlos con algun criterio (en este caso, de menor a mayor) y contar la cantidad de rachas, 
es decir, cuantas tiras de elementos del mismo conjunto se encuentran en la union ordenada de ambos.
El p-valor de este test consiste en contar la probabilidad de que haya dado dicha cantidad de runs o menor. El 
estadistico tiene una formula analitica, para una dada cantidad de runs se puede calcular su probabilidad, con 
esperanza y varianza conocidad, para hacer mas eficiente el calculo se aproximo su distribucion por una gaussiana 
(de esperanza y varianza correspondientes al de runs). Por eso solo funciona con conjuntos relativamente grandes, para
que se cumpla el teorema central del limite."""

#la funcion toma como parametro dos conjuntos A, B y retorna el estadistico de runs
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
    #cuento la cantidad cambios de conjunto de A a B y viceversa
    runs=0
    for i in range(len(conjunto)):
        if i==0 or conjunto[i][1]==conjunto[i-1][1]:
            pass
        else:
            runs+=1
    #la cantidad de rachas es eso mas 1
    runs+=1
    return runs

#la funcion toma dos conjuntos A,B, su estadistico de runs, y retorna el p-valor
#el parametro 'title' es un string opcional, si su valor es distinto a None plotea la distribucion (gaussiana), 
#el estadistico, indica el p-valor con dos decimales, y guarda el grafico con el nombre indicado en 'title'
def PvalR(A,B,runs,title=None):
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
    if title:
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
    else:
        pass
    return pValue

#%%                         FUNCIONES PARA SCATTER CON AJUSTE, CHI2 Y TEST Z
###                             REGRESION LINEAL CON GRAFICOS
"""Esto es una regresion lineal como de las que estamos acostumbrados, genera
la recta que mejor describe los datos que tenemos minimizando la distancia
cuadratica en el eje y a la recta.
    Entradas
A: conjunto de respuestas iniciales del subgrupo 1
B: conjunto de respuestas finales del subgrupo 1
C: conjunto de respuestas iniciales del subgrupo 2
D: conjunto de respuestas finales del subgrupo 2
stringAB y stringCD son strings con lo que va en el legend de cada uno
title, xlab e ylab son el titulo y lo del eje x e y del grafico
    Salidas
slopeAB, interceptAB, std_errAB son los que salen de la regresion entre A y B
slopeCD, interceptCD, std_errCD son los que salen de la regresion entre C y D
"""
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
"""Este test nos da una idea de la bondad del ajuste de RegLineal a los datos. El
estadistico se genera sumando sobre todos los datos la distancia al cuadrado de
un dato a la curva teorica en ese mismo X(el valor esperado) dividido(pesado) por
el valor en Y de la curva teorica en ese X. La distribucion es chi2 con N-1 
grados de libertad. El pvalor nos da una idea de que tan bien describe la recta
a nuestros datos, cuanto mayor a la media de la distribucion sea el estadistico
tendremos un pvalor menor y por lo tanto con mas ganas descartaremos la H0 de que
nuestros datos estan bien descriptos por la recta.
    Entradas
A: conjunto de respuestas iniciales del subgrupo 1
B: conjunto de respuestas finales del subgrupo 1
C: conjunto de respuestas iniciales del subgrupo 2
D: conjunto de respuestas finales del subgrupo 2
stringAB y stringCD son strings con lo que va en el legend de cada uno
slopeAB, interceptAB, slopeCD, interceptCD son los parametros que salen de RegLineal
    Salidas
pvAB y TAB son el p-valor y el estadistico obtenido para el subgrupo 1
pvCD y TCD son el p-valor y el estadistico obtenido para el subgrupo 2
"""
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
    pvAB=1-chi2.cdf(TAB,df) #pvalor para AB
    pvCD=1-chi2.cdf(TCD,df) #pvalor para CD
    plt.vlines(TAB,0,max(chi2.pdf(x, df)), color='red',
               label = 'P-valor '+stringAB+': '+str(np.format_float_scientific(pvAB, precision=2)))
    plt.vlines(TCD,0,max(chi2.pdf(x, df)),color='green',
               label = 'P-valor '+stringCD+': '+str(np.format_float_scientific(pvCD, precision=2)))
    plt.legend(loc='upper left')
    plt.tight_layout()
    return pvAB, TAB, pvCD, TCD
###                                 TEST Z
""" Este test se usa para comparar las pendientes obtenidas en RegLineal. 
Asumiendo que los datos son lineales y que las pendientes obtenidas son una variable
aleatoria continua con distribucion N(slopeAB,std_errAB^2) y N(slopeCD,std\_errCD^2),
es decir, que consideramos que los errores de la pendiente calculados por cuadrados minimos
son gaussianos con IC 68% [slopeAB-std_err^2,slopeAB+std_err^2]. Dicho esto consideramos
H0: las medias de las distribuciones son iguales (las pendientes son iguales)
H1: las medias de las distribuciones son distintas (las pendientes son distintas)
Por lo tanto la distribucion del estadistico sera una N(0,std_errAB^2 + std_errCD^2)
pues H0 es que las medias son iguales. Asi el estadistico es 
U = (slopeCD-slopeAB)/sqrt[std_errAB^2+std_errCD^2].
Por ultimo el pvalor es el doble del area encerrada entre el estadistico y 
el infinito correspondiente por ser a dos colas.

    Entradas
slopeAB, std_errAB, slopeCD, std_errCD son los parametros que salen de RegLineal
    Salidas
pv y U son el p-valor a dos colas y el estadistico del test
"""

def testZ(slopeAB, std_errAB, slopeCD, std_errCD, title):
    plt.figure(figsize=(5,4))
    plt.title(title)
    sigma=np.sqrt(std_errAB**2+std_errCD**2)
    x = np.linspace(-10*sigma, 10*sigma, 100000)
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
    Gau = gauss(x)/sum(gauss(x))
    if U > 0:
        pv=0
        i=0
        while x[i]<=U:
            pv+=Gau[i]
            i+=1
        pv=2*(1-pv)
        #dibujo el area que va a ser la mitad del p-valor por ser a dos colas
        xx=np.linspace(U,x[-1],10000)
        plt.fill_between(xx, 0, gauss(xx),color='r', alpha=0.4)
    else:
        pv=0
        i=0
        while x[i]<=U:
            pv+=Gau[i]
            i+=1
        pv=2*pv
        #dibujo el area que va a ser la mitad del p-valor por ser a dos colas
        xx=np.linspace(x[1],U,10000)
        plt.fill_between(xx, 0, gauss(xx),color='r', alpha=0.4)
    patch_pv = mpatches.Patch(color='blue',
                              label='P-valor: '+str(np.format_float_scientific(pv, precision=2)),
                              alpha=0.4)
    patch_area = mpatches.Patch(color='red',
                                label='Area bajo la curva: '+str(np.format_float_scientific(pv/2, precision=2)),
                                alpha=0.4)
    plt.legend(handles=[patch_pv, patch_area], loc='upper left')
    plt.tight_layout()
    return pv, U
