# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:40:09 2024

@author: Usuario
"""
#%%
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
sns.set_theme(context='paper')#configuro el formato de graficos
def redondeo(numero,error):
    cifras_cignificaitvas=len(f"{err_ordenada_menor:.1g}".split('.')[1])
    redondeado=round(numero,cifras_cignificaitvas)
    return(redondeado)


#%% DATOS
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
R=8.31446261815324 #J/(mol*K)

Temps=[26.5,35.5,45.25,52,60] #ºC

concentraciones=np.array([0.01,0.04,0.06,.08,0.1,0.15,0.35,0.4,0.5,0.6,0.75])
PM_SDS=288.38 #g/mol
concentraciones=concentraciones*10/PM_SDS #pasar de %m/v a molal

k=[[28,111.4,160,217,279,403,653,703,775,908,1058],
   [32.92,132.8,194.4,261,334,491,800,867,951,1124,1309],
   [39.3,160.4,232,313,399,596,987,1065,1168,1381,1605],
   [43.3,178.7,257,347,447,679,1133,1219,1333,1591,1827],
   [48.7,201,289,389,501,782,1310,1409,1547,1836,2110]
] #micro Siemes /cm

# Valores a CC_bajas
Temps_baja=[10,15,20]
k_baja=[[28.9,74.3,106.5,139.5,172.8,232,416,451,515,581,677],
        [32,84.5,120.1,158.2,197.5,269,473,516,591,666,782],
        [35.18,94.37,134.5,176.7,220,306,527,576,661,746,878]
]

#junto las dos series de mediciones
Temps=Temps_baja+Temps
k=k_baja+k
#TODO: Completar Errores de las mediciones
error_temp=0.2
error_concen=10*0.001/PM_SDS #propacación de error suponiendo Error_cc=10% de la cc mínima
error_conduct=1 #susuce 1 pero tendriamos que haber leido el manual del conductimetro, debe ser el 1%

#exporto datos para la corrección
columna_k=[]
for i in Temps:
    columna_k.append(f"k_T{i}C")
df={'Concentracion[m]':concentraciones}
df_={columna_k[i]:k[i] for i in range(len(Temps))}
df={**df,**df_}
df=pd.DataFrame(df)
df.to_csv('reportes/datos.csv',index=False)
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

print(f"{'T[ºC]':<15}{'n':<15}{'lambda_Na[uS/(cm.molal)]':<15}")

for T in Temps:
    def f(x):
        """El resolvedor busca raizes así que T=f(n) lo paso a F(n,T)=f(n)-T=0 y 
        las raices son el n que busco para cada T
        """
        global T
        return -0.28*x+(5.3E5)*(x**(-7/3))+7.5-T

    n = fsolve(f, T) #f=funcion a buscar raiz, T=valor cerce del que buscar
    
    lambda_na= 22.24+1.135*(T**3)
    lambda_na_lista.append(lambda_na)
    n_lista.append(n[0])

    print(f"{T:<15}{n[0]:<15.8g}{lambda_na:<15.8g}")
print("%"*60,"\n")

#Reporte
df_reporte=pd.DataFrame({'T[ºC]':Temps,'n':n_lista,'lambda_Na':lambda_na_lista})
#%% K vs Cs
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
DEBUG_K_VS_CS=False
p1_lista=[] #lista con las pendientes a Cs baja
p2_lista=[] #lista con las pendientes a Cs alta
o1_lista=[] #lista con las ordenadas a Cs baja
o2_lista=[] #lista con las ordenadas a Cs alta
cmc_lista=[] #lista con las a CMC

print(f"{'T':<8}{'CMC':<15}{'pendiente1':<15}{'ordenada1':<15}{'pendiente2':<15}{'ordenada2':<15}")
print(f"{'[ºC]':<8}{'[molal]':<15}{'uS/cm.molal':<15}{'uS/cm':<15}{'uS/cm.molal':<15}{'uS/cm':<15}")

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
    
    p1_lista.append(pendiente1[0])
    o1_lista.append(ordenada1)
    p2_lista.append(pendiente2[0])
    o2_lista.append(ordenada2)

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
    
    cmc_lista.append(CMC[0])

    print(f"{Temps[i]:<8}{CMC[0]:<15.8g}{pendiente1[0]:<15.8g}{ordenada1:<15.8g}{pendiente2[0]:<15.8g}{ordenada2:<15.8g}")
print('\n','%'*20,'\n')

#Reporte
df_reporte['CMC[molal]']=cmc_lista
df_reporte['pendiente1[uS/(cm.molal)]']=p1_lista
df_reporte['ordenada1[molal]']=o1_lista
df_reporte['pendiente2[uS/(cm.molal)]']=p2_lista
df_reporte['ordenada2[molal]']=o2_lista
#%% Figuras K vs Cs
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#colores
colores=sns.color_palette("colorblind", len(Temps))

fig,axes = plt.subplots(nrows=1,ncols=2)

for cuadrante in [0,1]:
    for i in range(len(Temps)):
        puntos=sns.scatterplot(
            x=concentraciones,
            y=k[i],
            label=f"T={Temps[i]}ºC",
            color=colores[i],
            ax=axes[cuadrante]
        )
        
        #el ajuste lineal a Cs baja
        cc_baja=[concentraciones[0],cmc_lista[1]*1.1]
        y_baja=np.array(cc_baja)*p1_lista[i]+o1_lista[i]

        ajuste_1=axes[cuadrante].plot(
            cc_baja,
            y_baja,
            c=colores[i],
            linestyle='--'
        )
        #el ajuste lineal a Cs alta
        cc_alta=[cmc_lista[i]*0.9,concentraciones[-1]]
        y_alta=np.array(cc_alta)*p2_lista[i]+o2_lista[i]

        ajuste_2=axes[cuadrante].plot(
            cc_alta,
            y_alta,
            c=colores[i],
            linestyle='--'
        )
        axes[cuadrante].set(
            xlabel="Cs [m]",
            ylabel=r"$\kappa \left[\mu S\right]$"
        )
#plt.vlines()
axes[1].vlines(
        x=np.array(cmc_lista),
        ymin=[0]*len(cmc_lista),
        ymax=np.array(cmc_lista)*np.array(p1_lista)+np.array(o1_lista),
        colors=colores
    )

plt.legend(bbox_to_anchor=(1,1),loc='upper left')
axes[0].legend_=None
axes[1].set(
    xlim=[0.0059,0.0073],
    ylim=[200,1010],
    ylabel=None,
    xticks=cmc_lista,
    xlabel="$Cs \ [m10^{-3}]$"
    )
zoom_x_label=[f"{i:.4g}" for i in np.array(cmc_lista)*1e3]
zoom_x_label[4]=None
axes[1].set_xticklabels(zoom_x_label,rotation=90)
plt.savefig('figuras/kvsCs.png',dpi=300,bbox_inches='tight')

#%% CALCULO ALPHA
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Y=a*X^2+b*X+c
# 0<alfa<1  es grado de ionizacion == relacion entre ionizado y no ionizado
#TODO:Para las ultimas 2 T la cuadrática no tiene raiz,
#revisé y estan bien los valores 
print(f"{'T[ºC]':<8}{'alfa<0':<15}{'alfa>0':<15}{'-b+4ac':<15}")
alpha_lista=[]
for i in range(len(Temps)):
    n=n_lista[i]
    p1=p1_lista[i]
    p2=p2_lista[i]
    lambda_na=lambda_na_lista[i]
    a=n**(2/3)*(p1-lambda_na)
    b=lambda_na
    c=-p2
    alpha=[(-b+np.sqrt(b**2-4*a*c))/(2*a),(-b-np.sqrt(b**2-4*a*c))/(2*a)]
    #elijo el alfa>0
    imprimo=False #variable para imprimir tabla
    for j in alpha:
        if j>0:
            alpha_lista.append(j)
            pos=j
            imprimo=True
        else:
            alfa_neg=j
    if imprimo==True:
        print(f"{Temps[i]:<8}{pos:<15.8g}{alfa_neg:<15.8g}{b**2-4*a*c:<15.8g}")
    else:
        print(f"{Temps[i]:<8}{'!∃ raiz real':<30}{b**2-4*a*c:<15.8g}")

#Reporte
df_reporte['alfa']=alpha_lista+[np.nan]*2
 #%% Grafico de la cuadratica, me importan los alfas<1
#Figura
fig,ax =plt.subplots()
minimox,maximox=-1,0.5
for i in range(len(Temps)):
    n=n_lista[i]
    p1=p1_lista[i]
    p2=p2_lista[i]
    lambda_na=lambda_na_lista[i]
    def alpha(x):
        global n
        global p1
        global p2
        global lambda_na
        return n**(2/3)*x**2*(p1-lambda_na)+x*lambda_na-p2

    x=np.linspace(minimox,maximox,100)
    y=alpha(x)
    sns.lineplot(
        x=x,
        y=y,
        zorder=2,
        label=f"T={Temps[i]}"
    ).set(
        xlim=[minimox,maximox],
        ylim=[-.25E6,.25E6]
    )
    plt.hlines(xmin=minimox,xmax=maximox,y=0,color='grey',zorder=1,linewidth=0.2)
ax.set(
    title=r"$f(n)=n^{2/3}\alpha^2(p_1-\lambda^{Na^+}+\alpha\lambda^{Na^+})$",
    xlabel=r'$\alpha$'
)
plt.legend(bbox_to_anchor=(1.01,1.05), loc="upper left")
plt.savefig('figuras/alfa_cuadrativa.png',dpi=300,bbox_inches='tight')
    
#%% CALCULO Gomic
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#
#    A PARTIR DE AHORA USO [T]=K
Temps=np.array(Temps)+273
alpha_lista=np.array(alpha_lista)
cmc_lista=np.array(cmc_lista)
def Go(T,alfa,CMC):
    global R
    return R*T*(2-alfa)*np.log(CMC)
Go_lista=[]

Go_lista=Go(
    T=Temps[:-2],
    alfa=alpha_lista,
    CMC=cmc_lista[:-2]
)
#quite las experiencias a T donde no pude calcular alfa
# for i in range(len(Temps[:-2])):
#     delta_Go_mic=Go(
#         T=Temps[i],
#         alfa=alpha_lista[i],
#         CMC=cmc_lista[i]
#         )
#     Go_lista.append(delta_Go_mic)
#Reporte
df_reporte['delta_Go[kJ/mol]']=np.append(Go_lista,[np.nan]*2)/1000
#%% alfa vs T
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#hay 2 ajustes linials, alta a T bajos y alfa a T altos
#el de menor T
# pendiente=dalfa/dT=[1/T]
medicion_cambio_pendiente=4
x= np.array(Temps[:medicion_cambio_pendiente])
y = np.array(alpha_lista[:medicion_cambio_pendiente])
x = sm.add_constant(x)
model_menor = sm.OLS(y,x).fit()
predictions_menor = model_menor.predict(x) 
ordenada_menor,pendiente_menor= model_menor.params
err_ordenada_menor, err_pendiente_menor = model_menor.bse
print_model_menor = model_menor.summary()
print(print_model_menor)

#el de mayor T
#el -2 es porque en las últimas 2 temp no pude calcular alfa
x= np.array(Temps[medicion_cambio_pendiente-1:-2])
y = np.array(alpha_lista[medicion_cambio_pendiente-1:])
x = sm.add_constant(x)
model_mayor = sm.OLS(y,x).fit()
predictions_mayor = model_mayor.predict(x) 
print_model_mayor = model_mayor.summary()
ordenada_mayor, pendiente_mayor= model_mayor.params
err_ordenada_mayor, err_pendiente_mayor = model_mayor.bse


#voy a tener 2 pendintes dalfa/dT, una para T<25 y otra pra T>25
#los pongo en la lista dalfa_dT_lista que está emparejada con la Temps
dalfa_dT_sub25=pendiente_menor
err_dalfa_dT_sub25=err_pendiente_menor
dalfa_dT_sob25=err_pendiente_mayor
err_dalfa_dT_sob25=err_pendiente_mayor

dalfa_dT_lista=np.array([dalfa_dT_sub25]*3+[dalfa_dT_sob25]*3)
print(print_model_menor)

#Reportes
# file_name=f"reportes/ajuste/alfavst_tmayor_{i}"
    # print_model_mayor.tables[i].as_csv
#%%
fig,axes = plt.subplots()
sns.scatterplot(
    x=Temps[:-2],
    y=alpha_lista
)
plt.plot(
    Temps[:medicion_cambio_pendiente],
    predictions_menor,
    ls='--',
    #label=r'$\alpha=$'+f'{redondeo(pendiente_menor,err_pendiente_menor)}T{redondeo(ordenada_menor,err_ordenada_menor)}'
)
plt.plot(
    Temps[medicion_cambio_pendiente-1:-2],
    predictions_mayor,
    ls='--',
    #label=r'$\alpha=$'+f"{pendiente_mayor}{err_pendiente_mayor}T{ordenada_menor}{err_ordenada_menor}'"
)
axes.set(
    xlabel="T[ºC]",
    ylabel=r"$\alpha$"
)
plt.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.savefig('figuras/alfavst.png',dpi=300,bbox_inches='tight')
#%% ln(CMC) vs T
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def pol_CMC(T,A,B,C):
    #el ajuste de la guia
    return A+B*T+C/T
def dlncmc_dt(T,A,B,C):
    #d/dT del ajuste
    return B-C/T**2
ln_cmc=np.log(cmc_lista)

param_pol_CMC,covar_pol_CMC= curve_fit(pol_CMC,Temps[3:],ln_cmc[3:])

dlnCMC_dT_lista = dlncmc_dt(np.array(Temps),*param_pol_CMC)


fig,axes=plt.subplots()

sns.scatterplot(
    x=Temps[:3],
    y=ln_cmc[:3],
    facecolor='red',
    zorder=10,
    label='datos descartados'
)
sns.scatterplot(
    x=Temps,
    y=ln_cmc
)
plt.plot(
    np.linspace(Temps[3],Temps[-1],20),
    pol_CMC(np.linspace(Temps[0],Temps[-1],20),*param_pol_CMC),
    linestyle='--',
    label=r"$CMC=A+BT+C/T$"+f"\nA={param_pol_CMC[0]:.4g} B={param_pol_CMC[1]:.4g} C={param_pol_CMC[2]:.4g}"
)
axes.set(
    xlabel="T [ºC]",
    ylabel="ln(CMC)"
)
plt.legend()
plt.savefig('figuras/lnCMCvsT.png',dpi=300,bbox_inches='tight')
#%% H y S
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def calculo_H(T:np.ndarray,alpha:np.ndarray,dlncmc_dt:np.ndarray,dalfa_dT:float,ln_cmc:np.ndarray):
    global R
    H= -R*T**2*((2-alpha)*dlncmc_dt-dalfa_dT*ln_cmc)
    return H
def calculo_S(T:np.ndarray,H:np.ndarray,G:np.ndarray):
    S= (H-G)/T 
    return S


H_lista=calculo_H(
    T=np.array(Temps)[:-2],
    alpha=np.array(alpha_lista),
    dlncmc_dt=dlnCMC_dT_lista[:-2],
    dalfa_dT=dalfa_dT_lista,
    ln_cmc=ln_cmc[:-2]
)

S_lista=calculo_S(
    T=np.array(Temps)[:-2],
    G=Go_lista,
    H=H_lista
)


print(f"{'Temp[ºC]':>15}{'dG':>15}{'dH':>15}{'dS':>15}")
for i in range(len(Temps[:-2])):
    print(f"{Temps[i]:>15}{Go_lista[i]:>15.8g}{H_lista[i]:>15.8g}{S_lista[i]:>15.8g}")
# %% Figura termo vs T
fig,axes = plt.subplots(nrows=1,ncols=2,sharey=True)
#lo paso a kJ en vez de joules
Go_lista*=1e-3
H_lista*=1e-3
S_lista*=1e-3
#nuestros datos
gibss=sns.lineplot(
    x=Temps[3:-2],
    y=Go_lista[3:],
    dashes=False,
    marker='o',
    ax=axes[0],
    label=r'$\Delta_{mic}G^0$'
)
sns.lineplot(
    x=Temps[3:-2],
    y=H_lista[3:],
    dashes=False,
    marker='o',
    ax=axes[0],
    label=r'$\Delta_{mic}H^0$'
)
sns.lineplot(
    x=Temps[3:-2],
    y=S_lista[3:]*Temps[3:-2],
    dashes=False,
    marker='o',
    ax=axes[0],
    label=r'$T\Delta_{mic}S^0$'
)

#datos del excel
gibss=sns.lineplot(
    x=Temps[:3],
    y=Go_lista[:3],
    dashes=False,
    marker='o',
    ax=axes[1],
    label=r'$\Delta_{mic}G^0$'
)
sns.lineplot(
    x=Temps[:3],
    y=H_lista[:3],
    dashes=False,
    marker='o',
    ax=axes[1],
    label=r'$\Delta_{mic}H^0$'
)
sns.lineplot(
    x=Temps[:3],
    y=S_lista[:3]*Temps[:3],
    dashes=False,
    marker='o',
    ax=axes[1],
    label=r'$T\Delta_{mic}S^0$'
)

plt.legend(bbox_to_anchor=(1,1),loc='upper left')
axes[0].legend_=None
axes[0].set(
    xlabel="T[K]",
    ylabel=r"$\Delta_{mis}G^0,\Delta_{mis}H^0,T\Delta_{mis}S^0\ \left[kJ/mol\right]$"
)
axes[1].set(
    xlabel="T[K]"
)
plt.savefig('figuras/TermovsT.png',dpi=300,bbox_inches='tight')
# %%


