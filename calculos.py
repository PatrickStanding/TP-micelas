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
#from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
#from my_libs import exportador_latex as etex
from scipy.optimize import leastsq
import sympy as sp

sns.set_theme(context='paper')#configuro el formato de graficos


def Prinvalor(valor,error):
    if 'e' in f"{error:.1g}":
        return f"{valor:.2g} pm {error:.1g}"
    else:
        return f"{valor:.5g} pm {error:.1g}"
        
def Reporte_valor(valor_lista,error_lista):
    return [(valor_lista[i],error_lista[i]) for i in range(len(valor_lista))]
def Plot_CI(X:list,CI_lower:list,CI_upper:list,Color='gray',Transparencia=0.4,Label=''):
    #plt.plot(X,CI_upper,color=Color)
    #plt.plot(X,CI_lower,color=Color)
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

## Valores a CC_bajas
#Temps_baja=[10,15,20]
#k_baja=[[28.9,74.3,106.5,139.5,172.8,232,416,451,515,581,677],
#        [32,84.5,120.1,158.2,197.5,269,473,516,591,666,782],
#        [35.18,94.37,134.5,176.7,220,306,527,576,661,746,878]
#]
#
##junto las dos series de mediciones
#Temps=Temps_baja+Temps
#k=k_baja+k
error_temp=0.02
error_concen=0.001/PM_SDS 
error_conduct=0.1 

err_Temps=[error_temp]*(len(Temps))


err_k= [error_conduct]*len(Temps)


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
print(f"{'T[ºC]':<15}{'n':<15}{'lambda_Na[uS/(cm.molal)]':<15}")

for T in Temps:
    def f(x):
        """El resolvedor busca raizes así que T=f(n) lo paso a F(n,T)=f(n)-T=0 y 
        las raices son el n que busco para cada T
        """
        global T
        return -0.28*x+(5.3E5)*(x**(-7/3))+7.5-T

    n,err_n = leastsq(func=f,x0=T) #f=funcion a buscar raiz, T=valor al partir del cual itera posibles soluciones
    
    lambda_na= 22.24+1.135*T
    err_lambda_na= abs(1.135)*error_temp
    
    lambda_na_lista.append(lambda_na)
    n_lista.append(n[0])
    err_n_lista.append(err_n)
    err_lambda_na_lista.append(err_lambda_na)

    prinn=f"{n[0]:.6g} pm {err_n:.1g}"
    prinlambda=f"{lambda_na:.8g} pm {err_lambda_na:.1g}"
    print(f"{T:<15}{prinn:<15}{prinlambda:<15}")
print("%"*60,"\n")

#Reporte
df_reporte=pd.DataFrame({
    'T[ºC]':Temps,
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
r2_menor_lista=[] #lista con los R²
r2_mayor_lista=[]
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


print(f"{'T':<8}{'CMC':<20}{'R2':<10}{'pendiente1':<20}{'ordenada1':<20}{'R2':<10}{'pendiente2':<20}{'ordenada2':<20}")
print(f"{'[K]':<8}{'[umolal]':<20}{'':<10}{'uS/cm.molal':<20}{'uS/cm':<20}{'':<10}{'uS/cm.molal':<20}{'uS/cm':<20}")

for i in range(len(k)):
    #ajuste lineal para Cs<CMC
    medicion_de_corte=6 #divide los datos en 2 series de 6 valores
    x= np.array(concentraciones[:medicion_de_corte])
    y = np.array(k[i][:medicion_de_corte])
    x = sm.add_constant(x)

    model1 = sm.OLS(y,x).fit()
    ordenada1,pendiente1= model1.params
    err_ordenada1, err_pendiente1= model1.bse
    r2_menor=model1.rsquared

    frame = model1.get_prediction(x).summary_frame(alpha=0.01)
    ci_1_low=frame.mean_ci_lower
    ci_1_up=frame.mean_ci_upper
    #print_model1= model1.summary()
    #print(print_model1)

    #ajuste lineal para Cs>CMC
    x = np.array(concentraciones[medicion_de_corte:])
    y = np.array(k[i][medicion_de_corte:])
    x = sm.add_constant(x)

    model2 = sm.OLS(y,x).fit()
    ordenada2,pendiente2= model2.params
    err_ordenada2, err_pendiente2= model2.bse
    r2_mayor= model2.rsquared

    print_model2= model2.summary()
    frame = model2.get_prediction(x).summary_frame(alpha=0.01)
    ci_2_low=frame.mean_ci_lower
    ci_2_up=frame.mean_ci_upper
    #print(print_model2)
    
    p1_lista.append(pendiente1)
    o1_lista.append(ordenada1)
    p2_lista.append(pendiente2)
    o2_lista.append(ordenada2)
    r2_menor_lista.append(r2_menor)
    r2_mayor_lista.append(r2_mayor_lista)
    err_p1_lista.append(err_pendiente1)
    err_o1_lista.append(err_ordenada1)
    err_p2_lista.append(err_pendiente2)
    err_o2_lista.append(err_ordenada2)
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

    print(f"{Temps[i]:<8}{Prinvalor(CMC*1e3,err_cmc*1e3):<20}{r2_menor:<10.3f}{Prinvalor(pendiente1,err_pendiente1):<20}{Prinvalor(ordenada1,err_ordenada1):<20}{r2_mayor:<10.3g}{Prinvalor(pendiente2,err_pendiente2):<20}{Prinvalor(ordenada2,err_ordenada2):<20}")
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


axes[1].vlines(
        x=np.array(cmc_lista),
        ymin=[0]*len(cmc_lista),
        ymax=np.array(cmc_lista)*np.array(p1_lista)+np.array(o1_lista),
        colors=colores
    )

plt.legend(bbox_to_anchor=(1,1),loc='upper left')
axes[0].legend_=None

axes[0].set(
    title='(1)',
    xticks=np.arange(0,0.03,0.005),
    xticklabels=np.arange(0,0.03,0.005)*1000,
    ylabel=r"$\kappa \left[\mu S\right]$",
    xlabel=r"$Cs \ [m mol/Kg]$"
    )

zoom_ofset=0.007 #porcentaje
axes[1].set(
    title='(2)',
    xlim=[min(cmc_lista)*(1-zoom_ofset),max(cmc_lista)*(1+zoom_ofset)],
    ylim=[200,1010],
    ylabel=None,
    xticks=cmc_lista,
    xlabel="$Cs \ [mmol/Kg]$"
    )
zoom_x_label=[f"{i:.4g}" for i in np.array(cmc_lista)*1e3]
zoom_x_label[0]+='\n'
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
    fmt='x',
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

ci_1=Plot_CI(
    X=concentraciones[:6],
    CI_lower=ci1_low_list[i],
    CI_upper=ci1_up_list[i],
    Transparencia=0.4
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
ci_1=Plot_CI(
    X=concentraciones[6:],
    CI_lower=ci2_low_list[i],
    CI_upper=ci2_up_list[i],
    Transparencia=0.2
)
axes.set(
    xlabel="Cs $[\mu m]$",
    ylabel=r"$\kappa \ [ \mu S / cm]$"
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

        print(f"{Temps[i]:<8}{float(raiz_pos):<8.6g} pm {float(err_alpha):.3g}")
    else:
        print(f"{Temps[i]:<8}{'no raiz real':<15}")

#Reporte
df_reporte['alfa']= Reporte_valor(alpha_lista,err_alpha_lista)

#%%
#Figuras : la cuadratica, con raiz=alfa
fig,ax =plt.subplots()
minimox,maximox=-0.5,0.5
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
        ylim=[-7E4,9E4]
    )
    plt.hlines(xmin=minimox,xmax=maximox,y=0,color='grey',zorder=1,linewidth=0.2)
ax.set(
    title=r"$f(n)=n^{2/3}\alpha^2(p_1-\lambda^{Na^+}+\alpha\lambda^{Na^+})$",
    xlabel=r'$\alpha$'
)
plt.legend(bbox_to_anchor=(1.01,1.05), loc="upper left")
plt.savefig('figuras/alfa_cuadrativa.png',dpi=300,bbox_inches='tight')
#%%
fig,axes = plt.subplots()
tamaño_vertical=0.1
for i in range(len(alpha_lista)):
    plt.plot(alpha_lista[i],i,'o',zorder=20)
    plt.hlines(y=i,xmin=alpha_lista[i]-err_alpha_lista[i],xmax=alpha_lista[i]+err_alpha_lista[i])
    plt.vlines(x=alpha_lista[i]-err_alpha_lista[i],ymin=i-tamaño_vertical,ymax=i+tamaño_vertical)
    plt.vlines(x=alpha_lista[i]+err_alpha_lista[i],ymin=i-tamaño_vertical,ymax=i+tamaño_vertical)

axes.set(
    xlabel=r"$\alpha$",
    yticks=[0,1,2,3,4],
    yticklabels=Temps,
    ylabel="T [ºC]",
    xticks=alpha_lista
)
axes.set_xticklabels([round(i,3) for i in alpha_lista],rotation=-90)
#%%
# CALCULO Gomic
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#  A PARTIR DE AHORA USO [T]=K
Temps=Temps
Temps=np.array(Temps)+273
alpha_lista=np.array(alpha_lista)
cmc_lista=np.array(cmc_lista)
def Go(T,alfa,CMC):
    global R
    return R*T*(2-alfa)*np.log(CMC)

Go_lista=Go(
    T=Temps,
    alfa=alpha_lista,
    CMC=cmc_lista
)

#propago error
def ErrGo(Go,T,alfa,CMC,eT,ealfa,eCMC):
    #lo propagué  amano con derivadas parciales
    error = abs(Go/T)*eT + abs(Go/(2-alfa))*ealfa + abs(R*T*(2-alfa)/CMC)*eCMC
    return error


err_go_lista= ErrGo(
    Go=Go_lista,
    T=Temps,
    alfa=alpha_lista,
    CMC=cmc_lista,
    eT=np.array(err_Temps),
    ealfa=np.array(err_alpha_lista),
    eCMC=np.array(err_cmc_lista)
)


print(f"Go [J/mol]")
for i in range(len(Go_lista)):
    print(f"{Go_lista[i]:.4g} pm {err_go_lista[i]:.1g}")
df_reporte['delta_Go[kJ/mol]']= Reporte_valor(
    Go_lista/1000,
    err_go_lista/1000
    )
#%%
# alfa vsT
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# pendiente=dalfa/dT=[1/T]
x= Temps
y = alpha_lista
x = sm.add_constant(x)
model_menor = sm.regression.linear_model.WLS(y,x,weights=1/np.array(err_alpha_lista)).fit()
print_model_menor = model_menor.summary()
print(print_model_menor)

ordenada_menor,pendiente_menor= model_menor.params
err_ordenada_menor, err_pendiente_menor = model_menor.bse
r2_menor=model_menor.rsquared
predictions_menor=model_menor.predict(x)

dalfa_dT=pendiente_menor
err_dalfa_dT=err_pendiente_menor


x_ci_menor=np.linspace(Temps[0],Temps[-1],100)
frame = model_menor.get_prediction(sm.add_constant(x_ci_menor)).summary_frame(alpha=0.01)
ci_1_low=frame.mean_ci_lower
ci_1_up=frame.mean_ci_upper

#los pongo en la lista dalfa_dT_lista que está emparejada con la Temps

dalfa_dT_lista=np.array([pendiente_menor]*len(Temps))
err_dalfa_dT_lista=np.array([err_pendiente_menor]*len(Temps))
print(print_model_menor)


#Reportes

#TODO exportar los resultados del ajuste cuadrático
df_reporte["dalfa_dT[1/K]"]=Reporte_valor(
    dalfa_dT_lista,
    err_dalfa_dT_lista)
#%%
# Figura alfa vs T
fig,axes = plt.subplots()
plt.errorbar(
    x=Temps,
    y=alpha_lista,
    yerr=err_alpha_lista,
    fmt='x',
    zorder=20,
)

etiqueta_menor=f'Ajuste lineal a T bajas $R^2=${r2_menor:.3f}\t'
etiqueta_menor+=r"$\frac{\partial \alpha}{\partial T}=$"+f"{dalfa_dT:.1e}"
etiqueta_menor+=" "+r"$\pm$"+f"{err_dalfa_dT:.1g} "+r"$T^{-1}$"
plt.plot(
    Temps,
    predictions_menor,
    ls='--',
    zorder=20,
    label=etiqueta_menor
)

Plot_CI(
    X=x_ci_menor,
    CI_lower=ci_1_low,
    CI_upper=ci_1_up
)

axes.set(
    xlabel="T [K]",
    ylabel=r"$\alpha$"
)
plt.legend(#bbox_to_anchor=(1,1),
           loc='upper left')
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
err_lncmc_lista=abs(1/cmc_lista)*err_cmc_lista

param_pol_CMC,covar_pol_CMC= curve_fit(
    pol_CMC,
    Temps,
    ln_cmc,
    #sigma=err_lncmc_lista[3:],
    #absolute_sigma=True
    )
err_param_pol_CMC=np.sqrt(np.diag(covar_pol_CMC))

dlnCMC_dT_lista = dlncmc_dt(np.array(Temps),*param_pol_CMC)

#%%
# Figura ln(CMC) vs T
fig,axes=plt.subplots()

# plt.errorbar(
#     x=Temps[:3],
#     y=ln_cmc[:3],
#     yerr=err_lncmc_lista[:3],
#     fmt='o',
#     color='red',
#     zorder=10,
#     label='datos descartados'
# )
plt.errorbar(
    x=Temps,
    y=ln_cmc,
    yerr=err_lncmc_lista,
    fmt='o'
)
plt.plot(
    np.linspace(Temps[0],Temps[-1],20),
    pol_CMC(np.linspace(Temps[0],Temps[-1],20),*param_pol_CMC),
    linestyle='--',
    label=r"$ln(CMC)=A+BT+C/T$"+f"\nA={param_pol_CMC[0]:.4g}   B={param_pol_CMC[1]:.4g} "+r"$K^{-1}$"+f"   C={param_pol_CMC[2]:.4g} "+r"$K$"
)


#Plot_CI(
#    X=Temps,
#    CI_lower=pol_CMC(Temps,*(param_pol_CMC+err_param_pol_CMC)),
#    CI_upper=pol_CMC(Temps,*(param_pol_CMC-err_param_pol_CMC))
#)

axes.set(
    xlabel="T [K]",
    ylabel="ln(CMC)"
)

plt.legend()
plt.savefig('figuras/lnCMCvsT.png',dpi=300,bbox_inches='tight')
#%%
# H y S
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def calculo_H(T:np.ndarray,alpha:np.ndarray,dlncmc_dt:np.ndarray,dalfa_dT:float,ln_cmc:np.ndarray):
    global R
    H= -R*T**2*((2-alpha)*dlncmc_dt-dalfa_dT*ln_cmc)
    return H
def calculo_S(T:np.ndarray,H:np.ndarray,G:np.ndarray):
    S= (H-G)/T 
    return S

def ERROR_H(H,T,alfa,dlncmc_dt,dalfa_dt,lncmc,cmc,eT,ealfa,edlncmc_dt,edalfa_dt,ecmc):
    global R
    eH= abs(2*H/T)*eT
    eH+= abs(R*T**2*dlncmc_dt)*ealfa 
    eH+= abs(-R*T**2*(2-alfa))*edlncmc_dt
    eH+= abs(R*T**2*lncmc)*edalfa_dt
    eH+= abs(R*T**2*dalfa_dt/cmc)*ecmc
    return eH

def ERROR_S(T,H,G,eT,eH,eG):
    eS= eH/T
    eS+= eG/T
    eS+= abs((G-H)/(T**2))*eT 
    return eS

H_lista=calculo_H(
    T=Temps,
    alpha=alpha_lista,
    dlncmc_dt=np.array(dlnCMC_dT_lista),
    dalfa_dT=dalfa_dT_lista,
    ln_cmc=ln_cmc
)

err_H_lista= ERROR_H(
    H=H_lista,
    T=Temps,
    alfa=alpha_lista,
    dlncmc_dt=dlnCMC_dT_lista,
    dalfa_dt=dalfa_dT_lista,
    lncmc=ln_cmc,
    cmc=cmc_lista,
    eT=np.array(err_Temps),
    ealfa=np.array(err_alpha_lista),
    edlncmc_dt=(1/cmc_lista)*np.array(err_cmc_lista),
    edalfa_dt=np.array(err_dalfa_dT_lista),
    ecmc=np.array(err_cmc_lista)
)

S_lista=calculo_S(
    T=Temps,
    G=Go_lista,
    H=H_lista
)

err_S_lista=ERROR_S(
    T=Temps,
    G=Go_lista,
    H=H_lista,
    eT=err_Temps,
    eG=err_go_lista,
    eH=err_H_lista
)



# Lo paso todo a Kj/mol
Go_lista=Go_lista
Go_lista*=1e-3
err_go_lista=err_go_lista
err_go_lista*=1e-3
S_lista=S_lista
S_lista*=1e-3
err_S_lista=err_S_lista
err_S_lista*=1e-3
H_lista=H_lista
H_lista*=1e-3
err_H_lista=err_H_lista
err_H_lista*=1e-3


TxS_lista=Temps*S_lista
err_TxS_lista=Temps*err_S_lista+abs(S_lista)*err_Temps

print(f"{'Temp':<10}{'dG':<22}{'dH':<22}{'dS':<22}{'T.dS':<22}")
print(f"{'K':<10}{'KJ/mol':<22}{'KJ/mol':<22}{'KJ/K.mol':<22}{'KJ/mol':<22}")
for i in range(len(Temps)):
    print(f"{Temps[i]:<10}{Prinvalor(Go_lista[i],err_go_lista[i]):<22}{Prinvalor(H_lista[i],err_H_lista[i]):<22}{Prinvalor(S_lista[i],err_S_lista[i]):<22}{Prinvalor(TxS_lista[i],err_TxS_lista[i]):<22}")


Go=np.mean(Go_lista)
err_Go=np.std(Go_lista)

print(f"\nGo= {Go:.1f} pm {err_Go:.1g}")

#Reporte
df_reporte['dH[KJ/mol]']=Reporte_valor(H_lista,err_H_lista)
df_reporte['dS[KJ/K.mol]']=Reporte_valor(S_lista,err_S_lista)

# %%
# Figura termo vs T
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
fig,axes = plt.subplots()
plt.errorbar(
    x=Temps,
    y=Go_lista,
    yerr=err_go_lista,
    label=r'$\Delta_{mic}G^0$',
    zorder=20,
    fmt='-x',
    capsize=3
)

plt.errorbar(
    x=Temps,
    y=H_lista,
    yerr=err_H_lista,
    fmt='-x',
    capsize=3,
    label=r'$\Delta_{mic}H^0$',
    zorder=20
)

plt.errorbar(
    x=Temps,
    y=TxS_lista,
    yerr=err_TxS_lista,
    fmt='-x',
    capsize=3,
    label=r'$T \Delta_{mic}S^0$'
)


plt.legend(bbox_to_anchor=(1,1),loc='upper left')

axes.set(
    xlabel="T[K]",
    ylabel=r"$\Delta_{mis}G^0,\Delta_{mis}H^0,T\Delta_{mis}S^0\ \left[kJ/mol\right]$",
    ylim=[-30,30]
)
plt.savefig('figuras/TermovsT.png',dpi=300,bbox_inches='tight')

#%%
# Reportes (ignorar)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

df_reporte.to_csv('reportes/valores.csv')
tex='reportes/datos.txt'

# cmc=df_reporte*1e-3
# variable_name='cmc_'+df_reporte['T[K]'].astype(str)
# variable=df_reporte['CMC[molal]']
# etex.variable(
#     tex,
#     variable_name,
#     variable
# )

# %%
# Exportar datos a latex
def redondeo(numero,error,debug=False):
    if error<numero:#error<1:
        
        #busco en que decimal está la primer cifra siginificativa del error
        err_str=str(error).split('.')[1]
        decimal=0
        for i in err_str:
            if i!='0':
                break
                if debug==True:
                    print(i)
                decimal+=1
            decimal+=1
        decimal+=1
    
        return f"{np.round(numero,decimal)}"+r" $\pm$ "f"{np.round(error,decimal)}"
    else:
        return f"{numero:.2g}"+r" $\pm$ "f"{error:.1g}"

#%%
def Tabla_Latex(df:pd.DataFrame,label:str,caption:str,alineado='c',):
    n_row=len(df.axes[0])
    n_col=len(df.axes[1])
    columns=df.columns

    #las partes de la tabla
    pattern0='''\\begin{table}[]\n\\centering\n\\begin{tabular}{@{}'''+alineado*n_col
    pattern1='@{}}\n\\toprule\n'
    pattern2=''

    pattern3="\\end{tabular}\n\\label{tab:"+label+"}\n\\end{table}" 
    if caption!=None:
        pattern3= "\n\\end{tabular}\n\\caption{"+caption+"}\n\\label{tab:"
        pattern3+= label+"}\n\\end{table}"
    

    #las columnas de la tabla
    for i in range(n_col):
        if i==n_col-1:
            pattern2+=columns[i]+" \\\\ \\midrule\n"
        else:
            pattern2+=columns[i]+" & "

    #el contenido de la tabla 
    content=[]
    for row in range(n_row):
        row_content=''
        for columna in columns: 
            index=df.index[row]
            cell=df[columna].iloc[row]
            temp_row=str(cell)+" & "
            if columna==columns[-1]:
                temp_row=str(cell)+" \\\\"
            row_content+=temp_row
                
        content.append(row_content)
    
    
    table_body=''
    stop=len(content)-1
    for i in range(len(content)):
        table_body+=content[i]
        if i == stop:
            table_body+=" \\bottomrule"

        table_body+='\n'
    output=pattern0+pattern1+pattern2+table_body+pattern3
    print(output)
    
#%%
#tabla CMC

export_cmc=[redondeo(cmc_lista[i]*1000,err_cmc_lista[i]*1000) for i in range(len(cmc_lista))]
df_cmc=pd.DataFrame({
    'T [K]':Temps,
    'CMC [$\\frac{mg}{L}$]':export_cmc,

})

Tabla_Latex(df_cmc,label='cmc',caption="valores de CMC calculados")
# tabla termo
#%%
export_go=[redondeo(Go_lista[i],err_go_lista[i]) for i in range(len(Go_lista))]
export_H=[redondeo(H_lista[i],err_H_lista[i]) for i in range(len(H_lista))]
export_S=[redondeo(S_lista[i],err_S_lista[i]) for i in range(len(S_lista))]
export_TS=[redondeo(TxS_lista[i],err_TxS_lista[i]) for i in range(len(TxS_lista))]
export_termo=pd.DataFrame({
    "T [K]":Temps,
    "$\\Delta_{\\text{mic}}G^0 [kJ/mol]$":export_go,
    "$\\Delta_{\\text{mic}}H^0 [kJ/mol]$":export_H,
    "$\\Delta_{\\text{mic}}S^0 [kJ/mol.K]$":export_S,
    "$T \\Delta_{\\text{mic}}S^0 [kJ/mol]$":export_TS,

})

Tabla_Latex(export_termo,label='termo',caption='Parámetros termodinámicos calulados')