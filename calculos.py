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
from my_libs import exportador_latex as etex
from scipy.optimize import leastsq
import sympy as sp

sns.set_theme(context='paper')#configuro el formato de graficos
def Prinvalor(valor,error):
    return f"{valor:.5g}pm{error:.1g}"
def Reporte_valor(valor_lista,error_lista):
    return [(valor_lista[i],error_lista[i]) for i in range(len(valor_lista))]
def Plot_CI(X:list,CI_lower:list,CI_upper:list,Color='gray',Transparencia=0.4,Label=''):
    plt.plot(X,CI_upper,color=Color)
    plt.plot(X,CI_lower,color=Color)
    plt.fill_between(
        x=X,
        y1=CI_lower,
        y2=CI_upper,
        color=Color,
        alpha=Transparencia,
        label=Label
    )

#%% 
#DATOS
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
R=8.31446261815324 #J/(mol*K)

Temps=[26.5,35.5,45.25,52,60] #ºC

concentraciones=np.array([0.01,0.04,0.06,.08,0.1,0.15,0.35,0.4,0.5,0.6,0.75])
PM_SDS=288.38 #g/mºol
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
error_temp=0.02
error_concen=0.001/PM_SDS 
error_conduct=0.1 

err_Temps=[error_temp]*(len(Temps))


err_k= [0.1*np.array(i) for i in k]


#Reportes
columna_k=[]
for i in Temps:
    columna_k.append(f"k_T{i}C")
col1={'Concentracion[m]':concentraciones}
col2={columna_k[i]:k[i] for i in range(len(Temps))}
df={**col1,**col2}
df_reporte=pd.DataFrame(df)
#%%
# BUSCO  n y lambda en funcion de T
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
n_lista=[]
err_n_lista=[]
lambda_na_lista=[]
err_lambda_na_lista=[]
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

    n,err_n = leastsq(func=f,x0=T) #f=funcion a buscar raiz, T=valor cerce del que buscar
    
    lambda_na= 22.24+1.135*(T**3)
    err_lambda_na= abs(3*1.135*T**2)*error_temp
    
    lambda_na_lista.append(lambda_na)
    n_lista.append(n[0])
    err_n_lista.append(err_n)
    err_lambda_na_lista.append(err_lambda_na)

    prinn=f"{n[0]:.6g}pm{err_n:.1g}"
    prinlambda=f"{lambda_na:.8g}pm{err_lambda_na:.1g}"
    print(f"{T:<15}{prinn:<15}{prinlambda:<15}")
print("%"*60,"\n")

#Reporte
df_reporte=pd.DataFrame({
    'T[K]':Temps,
    'n':Reporte_valor(n_lista,err_n_lista),
    'lambda_Na[uS/cm.molal]':Reporte_valor(lambda_na_lista,err_lambda_na_lista)
})
#%% 
# K vs Cs
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
DEBUG_K_VS_CS=False
p1_lista=[] #lista con las pendientes a Cs baja
p2_lista=[] #lista con las pendientes a Cs alta
o1_lista=[] #lista con las ordenadas a Cs baja
o2_lista=[] #lista con las ordenadas a Cs alta
cmc_lista=[] #lista con las a CMC

#lista de errores
err_p1_lista=[] 
err_p2_lista=[] 
err_o1_lista=[] 
err_o2_lista=[] 
err_cmc_lista=[] 
ci1_low_list=[]
ci1_up_list=[]
ci2_low_list=[]
ci2_up_list=[]


print(f"{'T':<8}{'CMC':<19}{'pendiente1':<15}{'ordenada1':<15}{'pendiente2':<15}{'ordenada2':<15}")
print(f"{'[K]':<8}{'[umolal]':<19}{'uS/cm.molal':<15}{'uS/cm':<15}{'uS/cm.molal':<15}{'uS/cm':<15}")

for i in range(len(k)):
    #ajuste lineal para Cs<CMC
    medicion_de_corte=6 #divide los datos en 2 series de 6 valores
    x= np.array(concentraciones[:medicion_de_corte])
    y = np.array(k[i][:medicion_de_corte])
    x = sm.add_constant(x)

    model1 = sm.OLS(y,x).fit()
    ordenada1,pendiente1= model1.params
    err_ordenada1, err_pendiente1= model1.bse
    print_model1= model1.summary()

    frame = model1.get_prediction(x).summary_frame(alpha=0.01)
    ci_1_low=frame.mean_ci_lower
    ci_1_up=frame.mean_ci_upper
    #print(print_model1)

    #ajuste lineal para Cs>CMC
    x = np.array(concentraciones[medicion_de_corte:])
    y = np.array(k[i][medicion_de_corte:])
    x = sm.add_constant(x)

    model2 = sm.OLS(y,x).fit()
    ordenada2,pendiente2= model2.params
    err_ordenada2, err_pendiente2= model2.bse

    print_model2= model2.summary()
    frame = model2.get_prediction(x).summary_frame(alpha=0.01)
    ci_2_low=frame.mean_ci_lower
    ci_2_up=frame.mean_ci_upper
    #print(print_model2)
    
    p1_lista.append(pendiente1)
    o1_lista.append(ordenada1)
    p2_lista.append(pendiente2)
    o2_lista.append(ordenada2)
    err_p1_lista.append(pendiente1)
    err_o1_lista.append(ordenada1)
    err_p2_lista.append(pendiente2)
    err_o2_lista.append(ordenada2)
    ci1_low_list.append(ci_1_low)
    ci1_up_list.append(ci_1_up)
    ci2_low_list.append(ci_2_low)
    ci2_up_list.append(ci_2_up)
    
    if DEBUG_K_VS_CS==True: 
        #veo grafico por gráfico
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
    #propagación de errores
    variables=p1,p2,o1,o2=sp.symbols('p1,p2,o1,o2',complex=False)
    errores_variables=ep1,ep2,eo1,eo2=sp.symbols('ep1,ep2,eo1,eo2',complex=False)
    eq=(o1-o2)/(p1-p2)
    error=[eq.diff(i) for i in variables]
    error_propagado=0
    for i in range(len(variables)):
        error_propagado+=abs(error[i])*errores_variables[i]
    #mostrar la formula de propagacion de error
    #display(error_propagado)
    evaluar=dict(zip(variables+errores_variables,[pendiente1,pendiente2,ordenada1,ordenada2,err_pendiente1,err_pendiente2,err_ordenada1,err_ordenada2]))
    err_cmc=error_propagado.evalf(subs=evaluar)
    err_cmc=float(err_cmc)
    
    cmc_lista.append(CMC)
    err_cmc_lista.append(err_cmc)

    print(f"{Temps[i]:<8}{Prinvalor(CMC*1e-3,err_cmc*1e-3):<19}{pendiente1:<15.8g}{ordenada1:<15.8g}{pendiente2:<15.8g}{ordenada2:<15.8g}")
print('\n','%'*20,'\n')

#Reporte
df_reporte['CMC[molal]']=[(cmc_lista[i],err_cmc_lista[i]) for i in range(len(cmc_lista))]
df_reporte['cmc_pendiente1[uS/(cm.molal)]']=Reporte_valor(p1_lista,err_p1_lista)
df_reporte['cmc_ordenada1[molal]']=Reporte_valor(o1_lista,err_o1_lista)
df_reporte['cmc_pendiente2[uS/(cm.molal)]']=Reporte_valor(p2_lista,err_p2_lista)
df_reporte['cmc_ordenada2[molal]']=Reporte_valor(o2_lista,err_o2_lista)

#%%
# Figuras K vs Cs
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
            label=f"T={Temps[i]}K",
            color=colores[i],
            ax=axes[cuadrante]
        )
        
        #el ajuste lineal a Cs baja
        cc_baja=[concentraciones[0],cmc_lista[i]*1.1]
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

axes[1].vlines(
        x=np.array(cmc_lista),
        ymin=[0]*len(cmc_lista),
        ymax=np.array(cmc_lista)*np.array(p1_lista)+np.array(o1_lista),
        colors=colores
    )

plt.legend(bbox_to_anchor=(1,1),loc='upper left')
axes[0].legend_=None

axes[0].set(title='(1)')
axes[1].set(
    title='(2)',
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
#%%
# Figura una sola serie
i=3
fig,axes= plt.subplots()

puntos=plt.errorbar(
    x=concentraciones,
    y=k[i],
    yerr=err_k[i],
    fmt='o',
    label=f"T={Temps[3]}K",
)


#el ajuste lineal a Cs baja
cc_baja=[concentraciones[0],cmc_lista[3]*1.1]
y_baja=np.array(cc_baja)*p1_lista[i]+o1_lista[i]

ajuste_1=plt.plot(
    cc_baja,
    y_baja,
    c=colores[i],
    linestyle='--',
    label="ajuste lineal"
)

ci_baja=plt.plot(
    concentraciones[:6],
    ci1_low_list[i][:6],
    color='grey'
)
ci_alta=plt.plot(
    concentraciones[:6],
    ci1_up_list[i][:6],
    color='gray'
)


#el ajuste lineal a Cs alta
cc_alta=[cmc_lista[i]*0.9,concentraciones[-1]]
y_alta=np.array(cc_alta)*p2_lista[i]+o2_lista[i]

ajuste_2=plt.plot(
    cc_alta,
    y_alta,
    c=colores[i],
    linestyle='--'
)
ci_baja2=plt.plot(
    concentraciones[6:],
    ci2_low_list[i],
    color='grey'
)
ci_alta2=plt.plot(
    concentraciones[6:],
    ci2_up_list[i],
    color='gray',
    label= r"ci $\alpha=0.05$"
)
axes.set(
    xlabel="Cs [\mu m]",
    ylabel=r"$\kappa \ \left[\mu S \right / cm]$"
)


axes.set_xticks(np.linspace(0,0.025,6),np.linspace(0,0.025,6)*1e3)
plt.legend(bbox_to_anchor=(1,1),loc='upper left')

plt.savefig('figuras/kvsCs_solo.png',dpi=300,bbox_inches='tight')

#%% 
# CALCULO ALPHA
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Y=a*X^2+b*X+c
# 0<alfa<1  es grado de ionizacion == relacion entre ionizado y no ionizado

# Para las ultimas 2 T la cuadrática no tiene raiz

alpha_lista=[]
err_alpha_lista=[]
print(f"{'T[ºC]':<8}{'alfa>0':<15}")
variables= n,p1,p2,lambda_na = sp.symbols('n,p1,p2,lambda_na',real=True)
errores= e_n,e_p1,e_p2,e_lambda_na = sp.symbols('e_n,e_p1,e_p2,e_lambda_na',real=True)


a=n**(sp.Rational(2,3))*(p1-lambda_na)
b=lambda_na
c=-p2
raiz_p=-b+sp.sqrt(b**2-4*a*c)/(2*a)
raiz_n=-b-sp.sqrt(b**2-4*a*c)/(2*a)

#display(a,b,c,raiz_p,raiz_n)
for i in range(len(Temps)):
    sub_dict={
        n:n_lista[i],
        p1:p1_lista[i],
        p2:p2_lista[i],
        lambda_na:lambda_na_lista[i],
    }
    pos=(-b+((b**2)-(4*a*c))**(0.5))/(2*a)
    raiz=(b**2)-(4*a*c)
    if raiz.evalf(subs=sub_dict)>0:
        _a=a.evalf(subs=sub_dict)
        _b=b.evalf(subs=sub_dict)
        _c=c.evalf(subs=sub_dict)
        raiz_pos=(-_b+((_b**2)-(4*_a*_c))**(0.5))/(2*_a)
        
        #propago error
        evaluar=dict(zip(variables+errores_variables,[n_lista[i],p1_lista[i],p2_lista[i],lambda_na_lista[i],err_n_lista[i],err_p1_lista[i],err_p2_lista[i],err_lambda_na_lista[i]]))
        error=[pos.diff(i) for i in variables]
        error_propagado=0
        for j in range(len(variables)):
            error_propagado+=abs(error[j])*errores_variables[j]
        error_propagado.evalf()
        #mostrar la formula de propagacion de error
        #display(error_propagado)
        err_alpha=error_propagado.evalf(subs=evaluar)
        alpha_lista.append(float(raiz_pos))
        err_alpha_lista.append(float(err_alpha))

        print(f"{Temps[i]:<8}{float(raiz_pos):<8.6g} +/- {float(err_alpha):.3g}")
    else:
        print(f"{Temps[i]:<8}{'no raiz real':<15}")

#Reporte
df_reporte['alfa']= Reporte_valor(alpha_lista+[np.nan]*2,err_alpha_lista+[np.nan]*2)

#%%
#Figuras : la cuadratica, con raiz=alfa
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
    
#%%
# CALCULO Gomic
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#  A PARTIR DE AHORA USO [T]=K

Temps=np.array(Temps)+273
alpha_lista=np.array(alpha_lista)
cmc_lista=np.array(cmc_lista)
def Go(T,alfa,CMC):
    global R
    return R*T*(2-alfa)*np.log(CMC)

Go_lista=Go(
    T=Temps[:-2],
    alfa=alpha_lista,
    CMC=cmc_lista[:-2]
)

#propago error
def ErrGo(Go,T,alfa,CMC,eT,ealfa,eCMC):
    #lo propagué  amano con derivadas parciales
    error = abs(Go/T)*eT + abs(Go/(2-alfa))*ealfa + abs(R*T*(2-alfa)/CMC)*eCMC
    return error


err_go_lista= ErrGo(
    Go=Go_lista,
    T=Temps[:-2],
    alfa=alpha_lista,
    CMC=cmc_lista[:-2],
    eT=np.array(err_Temps)[:-2],
    ealfa=np.array(err_alpha_lista),
    eCMC=np.array(err_cmc_lista)[:-2]
)

df_reporte['delta_Go[kJ/mol]']= Reporte_valor(
    np.append(Go_lista,[np.nan]*2)/1000,
    np.append(err_go_lista,[np.nan]*2)/1000
    )
#%%
# alfa vsT
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#hay 2 ajustes linials, alta a T bajos y alfa a T altos
#el de menor T
# pendiente=dalfa/dT=[1/T]
medicion_cambio_pendiente=4
x= Temps[:medicion_cambio_pendiente]
y = alpha_lista[:medicion_cambio_pendiente]
x = sm.add_constant(x)
model_menor = sm.OLS(y,x).fit()
print_model_menor = model_menor.summary()
print(print_model_menor)

ordenada_menor,pendiente_menor= model_menor.params
err_ordenada_menor, err_pendiente_menor = model_menor.bse
r2_menor=model_menor.rsquared

predictions_menor = model_menor.predict(x) 
frame = model_menor.get_prediction(x).summary_frame(alpha=0.05)
ci_1_low=frame.mean_ci_lower
ci_1_up=frame.mean_ci_upper


#el de mayor T
#el -2 es porque en las últimas 2 temp no pude calcular alfa
x= np.array(Temps[medicion_cambio_pendiente-1:-2])
y = np.array(alpha_lista[medicion_cambio_pendiente-1:])
x = sm.add_constant(x)
model_mayor = sm.OLS(y,x).fit()
print_model_mayor = model_mayor.summary()

ordenada_mayor, pendiente_mayor= model_mayor.params
err_ordenada_mayor, err_pendiente_mayor = model_mayor.bse
r2_mayor=model_mayor.rsquared

predictions_mayor = model_mayor.predict(x) 

x_ci_mayor=np.linspace(x[0][1],x[-1][1],50)
frame = model_mayor.get_prediction(sm.add_constant(x_ci_mayor)).summary_frame(alpha=0.05)
ci_2_low=frame.mean_ci_lower
ci_2_up=frame.mean_ci_upper


#voy a tener 2 pendintes dalfa/dT, una para T<25 y otra pra T>25
#los pongo en la lista dalfa_dT_lista que está emparejada con la Temps
dalfa_dT_sub25=pendiente_menor
err_dalfa_dT_sub25=err_pendiente_menor
dalfa_dT_sob25=err_pendiente_mayor
err_dalfa_dT_sob25=err_pendiente_mayor

dalfa_dT_lista=np.array([dalfa_dT_sub25]*3+[dalfa_dT_sob25]*3)
err_dalfa_dT_lista=np.array([err_dalfa_dT_sub25]*3+[err_dalfa_dT_sob25]*3)
print(print_model_menor)

#Reportes

#TODO exportar los resultados del ajuste cuadrático
df_reporte["dalfa_dT[1/K]"]=Reporte_valor(
    np.append(dalfa_dT_lista,[np.nan]*2),
    np.append(err_dalfa_dT_lista,[np.nan]*2))
#%%
fig,axes = plt.subplots()
sns.scatterplot(
    x=Temps[:-2],
    y=alpha_lista,
    zorder=20,
)
plt.plot(
    Temps[:medicion_cambio_pendiente],
    predictions_menor,
    ls='--',
    zorder=20,
    label=f'Ajuste lineal a T bajas  $R^2=${r2_menor:.2g}'
)

Plot_CI(
    X=Temps[:4],
    CI_lower=ci_1_low,
    CI_upper=ci_1_up
)
plt.plot(
    Temps[medicion_cambio_pendiente-1:-2],
    predictions_mayor,
    ls='--',
    label=f'Ajuste lineal a T bajas  $R^2=${r2_mayor:.2g}'
)
Plot_CI(
    X=x_ci_mayor,
    CI_lower=ci_2_low,
    CI_upper=ci_2_up
)


axes.set(
    xlabel="T [ºC]",
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
    xlabel="T [K]",
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


print(f"{'Temp[K]':>15}{'dG':>15}{'dH':>15}{'dS':>15}")
for i in range(len(Temps[:-2])):
    print(f"{Temps[i]:>15}{Go_lista[i]:>15.8g}{H_lista[i]:>15.8g}{S_lista[i]:>15.8g}")
#Reporte
df_reporte['dH[KJ/mol]']=np.append(H_lista,[np.nan]*2)
df_reporte['dS[KJ/K.mol]']=np.append(S_lista,[np.nan]*2)
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
    title="[1]",
    xlabel="T[K]",
    ylabel=r"$\Delta_{mis}G^0,\Delta_{mis}H^0,T\Delta_{mis}S^0\ \left[kJ/mol\right]$"
)
axes[1].set(
    xlabel="T[K]",
    title="[2]"
)
plt.savefig('figuras/TermovsT.png',dpi=300,bbox_inches='tight')
# %%Reportes (ignorar)

df_reporte.to_csv('reportes/valores.csv')
tex='reportes/datos.txt'

cmc=df_reporte*1e-3
variable_name='cmc_'+df_reporte['T[K]'].astype(str)
variable=df_reporte['CMC[molal]']
etex.variable(
    tex,
    variable_name,
    variable
)

# %%
