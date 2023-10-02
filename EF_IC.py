import scipy as sc
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

"""Cálculos de Campo Elétrico, Corrente Iônica, Rádio Interferência e Ruído Audível - HVDC
João Vítor Drumond
2023

Passo a passo
1 - Calcular o gradiente para a condição climática especificada
3 - Obter o nível de saturação da região
4 - Calcular o campo elétrico livre de corona
5 - Calcular o campo elétrico saturado
6 - Calcular as densidades de corrente saturadas
7 - Calcular as densidades de corrente livres de corona

Obs.:
- Para que exista circulação de corrente iônica no meio, pressupõe-se
a condição de máxima saturação de corona. Sendo assim, para ambientes
livres de corona, a densidade de corrente iônica é nula.
- As equações do campo elétrico e da densidade de corrente iônica para um
ambiente com saturação de corona em função da distância na faixa para o 
eixo da LT somente podem ser utilizadas em um intervalo espacial
pré-definido, dado por: 1 < (x-P/2)/H < 4. Sendo assim, essa equação pode
ser aplicada no cálculo dos valores dessas grandezas no limite da faixa.
- Para o Ruído Audível máximo, adicionar 5dBA ao valor médio calculado.
Para o Ruído Audível sob chuva, subtrair 6dBA do valor médio calculado.

"""

#*****************************************************************

#                              FUNÇÕES 

#*****************************************************************


#Campo elétrico para um ambiente livre de corona (Valor mínimo)

def E_sc(V,H,deq,P,X):
    if np.size(X)>1:
        y = np.ndarray.tolist(((2*V*H)/(np.log((4*H)/deq)-(1/2)*np.log(((4*np.power(H,2))+np.power(P,2))/np.power(P,2))))*(((1/(np.power(H,2)+np.power(np.array(X)-(P/2),2))))-(1/(np.power(H,2)+np.power((np.array(X)+(P/2)),2)))))
    else:
        y = ((2*V*H)/(np.log((4*H)/deq)-(1/2)*np.log(((4*np.power(H,2))+np.power(P,2))/np.power(P,2))))*(((1/(np.power(H,2)+np.power(X-(P/2),2))))-(1/(np.power(H,2)+np.power((X+(P/2)),2))))
    return y

#Campo elétrico para um ambiente com saturação de corona (Valor máximo)
def E_c(V,H,P,X):
    k=1
    y=[]
    w=[]
    if(np.size(X)>1):
        for i in range(np.size(X)):
            if (1 < abs((X[i]-P/2)/H) and abs((X[i]-P/2)/H) <= 4):
                w.append(X[i])
                y.append(1.46*(1-np.exp(-2.5*(P/H)))*np.exp(-0.7*(X[i]-(P/2))/H)*(V/H))
                k = k+1
    elif(1 < abs((X-P/2)/H) and abs((X-P/2)/H) <= 4):        
        y = 1.46*(1-np.exp(-2.5*(P/H)))*np.exp(-0.7*(X-(P/2))/H)*(V/H)
    else:
        print(f"Limite Atingido!\n")
    
    return [w,y]

#Densidades de corrente iônica positiva para um ambiente com saturação de corona (Valor máximo) 
def Jps(V,H,P,X):
    k=1
    y=[]
    if np.size(X) > 1:
        for i in range(np.size(X)):
            if (1 < abs((X[i]-P/2)/H) and abs((X[i]-P/2)/H) < 4):
                y.append(1.54e-15*(1-np.exp(-1.5*(P/H)))*np.exp(-1.75*(X[i]-(P/2))/H)*(np.power(V,2)/np.power(H,3)))
                k=k+1
    elif(1 < abs((X-P/2)/H) and abs((X-P/2)/H) < 4):
        y = 1.54e-15*(1-np.exp(-1.5*(P/H)))*np.exp(-1.75*(X-(P/2))/H)*(np.power(V,2)/np.power(H,3))   
    else:
        print(f'Limite Atingido!')
    return y

#Densidades de corrente iônica negativa para um ambiente com saturação de corona (Valor máximo)
def Jns(V,H,P,X):
    k=1
    y=[]
    if np.size(X) > 1:
        for i in range(np.size(X)):
            if (1 < abs((X[i]-P/2)/H) & abs((X[i]-P/2)/H) < 4):
                y.append(2e-15*(1-np.exp(-1.5*(P/H)))*np.exp(-1.75*(X[i]-(P/2))/H)*(np.power(V,2)/np.power(H,3)))
                k=k+1
    elif(1 < abs((X-P/2)/H) and abs((X-P/2)/H) < 4):
        y = 2e-15*(1-np.exp(-1.5*(P/H)))*np.exp(-1.75*(X-(P/2))/H)*(np.power(V,2)/np.power(H,3))
    else:
        print(f'Limite Atingido!')
    return y

#*****************************************************************

#                              ROTINA 

#*****************************************************************

#-----------------------------------------------------------------
#                         DADOS DE ENTRADA 
#-----------------------------------------------------------------

D = 600e-3 #Diâmetro do bundle
n = 6 #Número de subcondutores
#d = 3.762e-2 #Diâmetro dos subcondutores do bundle, em cm
d = 3.69e-2
L = 114 #Largura da faixa de servidão, em m
P = 20.8 #Espaçamento entre polos, em  m
X = [x for x in range(0,int(L/2))] #Vetor distâncias ao longo da faixa, em m
f = 16.76 #Flecha do condutor (Flecha máxima creep), em m
#H = 19.9+(f/3);#Menor altura do cabo para o solo, em m
H = 21
V = 800e3 #Tensão da LT, em V
G0p = 9 #Fator climático de gradiente polo positivo (Verão - L50)
G0n = 9 #Fator climático de gradiente polo negaivo (Verão - L50)
Kp = 0.037 #Constante de saturação positiva
Kn = 0.015 #Constante de saturação negativa
frq = 1 #Frequência para cálculo de Rádio Interferência, em MHz
q = 500 #Altitude, em m

"""
mi_p = 1.5e-4;//Mobilidade iônica positiva
mi_n = 1.15e-4;//Mobilidade iônica negativa
ce = 1.6e-19;//Carga de um elétron
"""

#-----------------------------------------------------------------
#                CAMPO ELÉTRICO E CORRENTE IÔNICA 
#-----------------------------------------------------------------

#Diâmetro equivalente do bundle
if n == 1:
    deq = d
else: 
  deq = D*np.power(((n*d)/D),(1/n)) 

#Raio para o gradiente
R = (D/2)/np.sin(3.14159265359/n)

#Coeficiente do gradiente do condutor (em kV/cm / kV))
A = n*(d/2)
#B = (A*(R^(n-1)))^(1/n)
B = np.power(A*np.power(R,(n-1)),(1/n))
C = np.sqrt(np.power(((2*H)/P),2)+1)
#g = ((1+(n-1)*((d/2)/R))/(A*np.log((2*H)/(B*C))))/1e2
g = ((1+(n-1)*((d/2)/R))/(A*np.log((2*H)/(B*C))))/1e2

#Gradiente do condutor (em kV/cm)
G_cond = g*V/1e3

#Grau de saturação de corona
Sp = 1 - np.exp(-Kp*(G_cond-G0p))
Sn = 1 - np.exp(-Kn*(G_cond-G0n))

"""Campo elétrico sem corona"""
E_sc_max = max(E_sc(V,H,deq,P,X)) #Campo máximo
E_sc_cd = E_sc(V,H,deq,P,(P/2)) #Abaixo do condutor
E_sc_lf = E_sc(V,H,deq,P,(L/2)) #No limite da faixa

"""Campo elétrico para o ambiente com saturação de corona (Valor máximo)"""
E_c_max = 1.31*(1-np.exp(-1.7*(P/H)))*(V/H) #Valor máximo no meio da faixa
[lim_E, E_c_fai] = E_c(V,H,P,X) 
[vet,E_c_fs] = E_c(V,H,P,(L/2)) #Valor máximo no limite da faixa


"""Campo elétrico efetivo"""
#No meio da faixa
Ep_mf = E_sc_cd + Sp*(E_c_max - E_sc_cd) #Positivo
En_mf = -E_sc_cd + Sn*(-E_c_max + E_sc_cd) #Negativo

#No limite da faixa
Ep_lf = E_sc_lf + Sp*(E_c_fs - E_sc_lf) #Positivo
En_lf = -E_sc_lf + Sn*(-E_c_fs + E_sc_lf) #Negativo

#No intervalo de cálculo
Ep_int_1 = np.ndarray.tolist(np.array(E_sc(V,H,deq,P,lim_E)) + Sp*(np.array(E_c(V,H,P,lim_E)) - np.array(E_sc(V,H,deq,P,lim_E))))
EP_int = [Ep_int_1[1][u] if (u>=0 and u<np.size(Ep_int_1[1])) else Ep_int_1[0][u-np.size(Ep_int_1[0])] for u in X[0:2*np.size(Ep_int_1[0])]]

"""Densidades de corrente iônica para um ambiente com saturação de corona (Valor máximo)"""
#Positiva
Jps_mf = (1.65e-15)*(1-np.exp(-0.7*(P/H)))*(np.power(V,2)/np.power(H,3)) #Valor máximo no meio da faixa
Jps_lf = Jps(V,H,P,(L/2)) #Valor máximo no limite da faixa 

#/Negativa
Jns_mf = (-2.15e-15)*(1-np.exp(-0.7*(P/H)))*(np.power(V,2)/np.power(H,3)) #Valor máximo no meio da faixa
Jns_lf = Jns(V,H,P,(L/2)) #/Valor máximo no limite da faixa 

"""Densidades de corrente iônica efetivas"""
#Positivo
Jp_mf = Sp*Jps_mf #No meio da faixa
Jp_lf = Sp*Jps_lf #No limite da faixa
#Negativo
Jn_mf = Sn*Jns_mf #No meio da faixa
Jn_lf = Sp*Jns_lf #No limite da faixa

"""Resumo"""
print("-------------------------------------\nCAMPO ELÉTRICO E CORRENTE IÔNICA\n-------------------------------------\n")
print(f'***Campo positivo [kV/m]***\n')
print(f'No meio da faixa: {Ep_mf*1e-3}')
print(f'No limite da faixa: {Ep_lf*1e-3}')
print(f'\n***Campo negativo [kV/m]***\n')
print(f'No meio da faixa: {En_mf*1e-3}')
print(f'No limite da faixa: {En_lf*1e-3}')
print(f'\n***Densidade de corrente positiva [nA/m²]***\n')
print(f'No meio da faixa: {Jp_mf*1e9}')
print(f'No limite da faixa: {Jp_lf*1e9}')
print(f'\n***Densidade de corrente negativa [nA/m²]***\n')
print(f'No meio da faixa: {Jn_mf*1e9}')
print(f'No limite da faixa: {Jn_lf*1e9}\n')

"""Gráficos"""
"""plt.plot(X,EP_int)
plt.show()"""

#-----------------------------------------------------------------
#                         RÁDIO INTERFERÊNCIA 
#-----------------------------------------------------------------
print("-------------------------------------\n            R.I. E R.A.\n-------------------------------------\n")
#Raio equivalente do bundle
req = R*np.power((n*R),(1/n))

#Distância radial para o polo positivo
Dpp = np.sqrt(np.power(X,2)+np.power(H+1,2))

#Rádio Interferência ao longo da faixa
RI = 51.7 + 86*np.log10(G_cond/25.6) + 40*np.log10((d*100)/4.62)+10*(1-np.power(np.log10(10*frq),2)) + 40*np.log10(19.9/Dpp)+(q/300)

print(f'***Rádio Interferência [dB]***\n')
print(f'No meio da faixa: {np.max(RI)}')
print(f'No limite da faixa: {RI[int(L/2-1)]}\n')

"""Gráficos"""
#plt.plot(X,RI)
#plt.show()

#-----------------------------------------------------------------
#                           RUÍDO AUDÍVEL 
#-----------------------------------------------------------------

#Constantes empíricas
if n > 2:
    k = 25.6
    An0 = -100.62
else:
    k = 0
    An0 = -93.4

#Ruído Audível médio ao longo da faixa
An = An0 + 86*np.log10(G_cond) + k*np.log10(n) + 40*np.log10(d*100) - 11.4*np.log10(Dpp) + q/300

print(f'***Ruído Audível [dB]***\n')
print(f'No meio da faixa: {np.max(An)}')
print(f'No limite da faixa: {An[int(L/2-1)]}\n')

"""Gráficos"""
#plt.plot(X,An)
#plt.show()