# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:40:09 2024

@author: Usuario
"""
#%%
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
#%% DATOS
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Temps=[26.5,35.5,45.25,52,60] #ºC

concentraciones=np.array([0.01,0.04,0.06,.08,0.1,0.15,0.35,0.4,0.5,0.6,0.75])
PM_SDS=288.38 #g/mol
concentraciones=concentraciones*10/PM_SDS #pasar de %m/v a molal

k=[[28,111.4,160,217,279,403,653,703,775,908,1058],
   [32.92,132.8,194.4,261,334,491,800,867,951,1124,1309],
   [39.3,160.4,232,313,399,596,987,1065,1168,1381,1605],
   [43.3,178.7,257,347,447,679,1133,1219,1333,1591,1827],
   [48.7,201,289,389,501,782,1310,1409,1547,1836,2110]
]

#TODO: Completar Errores de las mediciones
error_temp=0.2
error_concen=10*0.001/PM_SDS #propacación de error suponiendo Error_cc=10% de la cc mínima
error_conduct=1 #susuce 1 pero tendriamos que haber leido el manual del conductimetro, debe ser el 1%
#%% BUSCO  n y lambda en funcion de T
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
n_lista=[]
lambda_na_lista=[]
def n_test(n,T):
    """Funcion de Debug
    Revisa que |T-T(n)|<1% para asegurarme que calulé bien el n a cada T
    """
    Tn= -0.28*n + (5.3E5)*(n**(-7/3)) + 7.5
    diff= abs(T-Tn)/T
    if diff<0.01:
        return print("BIEN, n(T={T})")
    else:
        return print("MAL, n(T={T})") 


for T in Temps:
    def f(x):
        """El resolvedor busca raizes así que T=f(n) lo paso a F(n,T)=f(n)-T=0 y 
        las raices son el n que busco para cada T
        """
        global T
        return -0.28*x+(5.3E5)*(x**(-7/3))+7.5-T

    n = fsolve(f, T)
    
    lambda_na= 22.24+1.135*T**3
    lambda_na_lista.append(lambda_na)
    n_lista.append(n)
    
    print(f"T={T}\tn={n}\tlambda_Na={lambda_na}")
print("%"*60,"\n")

#%% K vs Cs
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
DEBUG_K_VS_CS=False
p1_lista=[] #lista con las pendientes a Cs baja
p2_lista=[] #lista con las pendientes a Cs alta
o1_lista=[] #lista con las ordenadas a Cs baja
o2_lista=[] #lista con las ordenadas a Cs alta
cmc_lista=[] #lista con las a CMC

for i in range(len(k)):
    #ajuste lineal para Cs<CMC
    medicion_de_corte=6 #divide los datos en 2 series de 6 valores
    x= concentraciones[:medicion_de_corte].reshape((-1,1))
    y = np.array(k[i][:medicion_de_corte])
    model = LinearRegression()
    model.fit(x, y)
    pendiente1=model.coef_
    ordenada1=model.intercept_
    
    #ajuste lineal para Cs>CMC
    x2 = concentraciones[medicion_de_corte:].reshape((-1,1))
    y2 = np.array(k[i][medicion_de_corte:])
    model.fit(x2, y2)
    pendiente2=model.coef_
    ordenada2=model.intercept_
    
    p1_lista.append(pendiente1)
    o1_lista.append(ordenada1)
    p2_lista.append(pendiente2)
    o2_lista.append(ordenada2)

    print(f"TEMP {Temps[i]}")
    print(f"ordenada={ordenada1}")
    print(f"pendiente={pendiente1}")
    print(f"ordenada={ordenada2}")
    print(f"pendiente={pendiente2}")
    if DEBUG_K_VS_CS==True: 
        fig, axes = plt.subplots()
        data=plt.scatter(
                concentraciones,
                k[i]
                )
        ajuste1=plt.plot(
                [concentraciones[0],concentraciones[medicion_de_corte]],
                [concentraciones[0]*pendiente1+ordenada1,concentraciones[medicion_de_corte]*pendiente1+ordenada1]
                )
        ajuste2=plt.plot(
                [concentraciones[medicion_de_corte],concentraciones[-1]],
                [concentraciones[medicion_de_corte]*pendiente2+ordenada2,concentraciones[-1]*pendiente2+ordenada2]
                )
        axes.set(
            title=f"T={Temps[i]}"
        )
    
    CMC=abs(ordenada1-ordenada2)/abs(pendiente1-pendiente2)
    print(f"CMC={CMC}")
    print('\n','%'*20,'\n')
    
    cmc_lista.append(CMC[0])
#%% Figuras K vs Cs
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sns.set_theme(context='paper')

#colores
colores=sns.color_palette("colorblind", len(Temps))

fig,axes = plt.subplots()

for i in range(len(Temps)):
    puntos=sns.scatterplot(
        x=concentraciones,
        y=k[i],
        label=f"T={Temps[i]}ºC",
        color=colores[i]
    )
    
    #el ajuste lineal a Cs baja
    cc_baja=[concentraciones[0],concentraciones[medicion_de_corte+1]]
    y_baja=cc_baja*p1_lista[i]+o1_lista[i]

    ajuste_1=plt.plot(
        cc_baja,
        y_baja,
        c=colores[i],
        linestyle='--'
    )
    #el ajuste lineal a Cs alta
    cc_alta=[concentraciones[medicion_de_corte-1],concentraciones[-1]]
    y_alta=cc_alta*p2_lista[i]+o2_lista[i]

    ajuste_2=plt.plot(
        cc_alta,
        y_alta,
        c=colores[i],
        linestyle='--'
    )
    axes.set(
        title="$\kappa\ vs\ Cs$",
        xlabel="Cs [m]",
        ylabel=r"$\kappa \left[\mu S\right]$"
    )
plt.legend()
plt.savefig('figuras/kvsCs.png',dpi=300,bbox_inches='tight')

#%% CALCULO ALPHA
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Y=a*X^2+b*X+c
alpha_lista=[]
for i in range(len(Temps)):
    n=n_lista[i]
    p1=p1_lista[i]
    p2=p2_lista[i]
    lambda_na=lambda_na_lista[i]
    a=n**(2/3)*(p1-lambda_na)
    b=lambda_na
    c=-p2
    alpha=[(-b+np.sqrt(b**2-4*a*c))/(2*a),(-b+np.sqrt(b**2-4*a*c))/(2*a)]

#%% 
    plt.scatter(Temps,np.log(np.array(cmc_lista)))
    