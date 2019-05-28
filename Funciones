import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
