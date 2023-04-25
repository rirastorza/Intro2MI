#!/usr/bin/env python3
"""
Plotea resultados de primera medición experimental!

"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def agrupa(datos,nfrec,nant):
    """
    Agrupa los datos medidos de la siguiente forma (para nant = 8):
    #Med_P1P2  Med_P1P3  Med_P1P4  Med_P1P5  Med_P1P6  Med_P1P7  Med_P1P8  
    #Med_P2P3  Med_P2P4  Med_P2P5  Med_P2P6  Med_P2P7  Med_P2P8  Med_P1P2
    #Med_P3P4  Med_P3P5  Med_P3P6  Med_P3P7  Med_P3P8  Med_P1P3  Med_P2P3  
    #Med_P4P5  Med_P4P6  Med_P4P7  Med_P4P8  Med_P1P4  Med_P2P4  Med_P3P4
    #Med_P5P6  Med_P5P7  Med_P5P8  Med_P1P5  Med_P2P5  Med_P3P5  Med_P4P5
    #Med_P6P7  Med_P6P8  Med_P1P6  Med_P2P6  Med_P3P6  Med_P4P6  Med_P5P6
    #Med_P7P8  Med_P1P7  Med_P2P7  Med_P3P7  Med_P4P7  Med_P5P7  Med_P6P7
    #Med_P1P8  Med_P2P8  Med_P3P8  Med_P4P8  Med_P5P8  Med_P6P8  Med_P7P8
    """
    Sinci = np.zeros((len(datos[0,1:nant]),len(datos[0,1:nant])))
    
    nant_anterior = 1
    for i in range(nant-1):
        #print(i,nant-i-1,nant_anterior,nant_anterior+nant-i)
        Sinci[i,:nant-i-1] = datos[nfrec,nant_anterior:nant_anterior+nant-i-1] 
        nant_anterior = nant_anterior+nant-i-1
    #print(Sinci)
    for i in range(nant-1):
        rows, cols = np.indices((len(datos[0,1:nant]),len(datos[0,1:nant])))
        row_vals = np.diag(np.fliplr(rows), k=-i-1)
        col_vals = np.diag(np.fliplr(cols), k=-i-1)
        Sinci[row_vals, col_vals]= Sinci[i,0:nant-2-i]
        #print(Sinci[i,0:nant-1-i])
    
    return Sinci[:,:]



incidente = 'MED_NaCL_8pos_sin_teflon_Npos_8_04-Apr-2023_15h_11m_30s_pos_completo_col.txt'
canonico = 'MED_NaCL_8pos_teflon_Npos_8_04-Apr-2023_14h_44m_26s_pos_completo_col.txt'

datos = np.loadtxt(incidente,skiprows=1)
f_medida = 1e6*datos[:,0]
datos_canonico = np.loadtxt(canonico,skiprows=1,delimiter=',')

nfrec = 64
nant = 8

Sinc_FINAL = agrupa(datos,nfrec,nant)
S_FINAL = agrupa(datos_canonico,nfrec,nant)

########
#Comparación con experimento 4
nylon = "MED_NaCL_8pos_Nylon_offset_Npos_8_04-Apr-2023_15h_33m_38s_pos_completo_col.txt"
datos_cilindronylon = np.loadtxt(nylon,skiprows=1)

S_nylon_FINAL = agrupa(datos_cilindronylon,nfrec,nant)


fig10 = plt.figure(10)
ax1 = fig10.add_subplot(121)
ax1.plot(Sinc_FINAL[0,:],'ob-',label='Incidente, Tx: 1, f: '+str(f_medida[nfrec]/1e6)+' MHz')
#ax1.plot(Sinc_FINAL[1,:],'xr-',label='Incidente, Tx: 2')
#ax1.plot(Sinc_FINAL[2,:],'sg-',label='Incidente, Tx: 3')
#ax1.plot(Sinc_FINAL[3,:],'vy-',label='Incidente, Tx: 4')
#ax1.plot(Sinc_FINAL[6,:],'^m-',label='Incidente, Tx: 7')
#ax1.plot(S_FINAL[0,:],'ob--',label='Teflon, Tx: 1, f: '+str(f_medida[nfrec]/1e6)+' MHz')
ax1.plot(S_nylon_FINAL[0,:],'ob-.',label='Nylon, Tx: 1, f: '+str(f_medida[nfrec]/1e6)+' MHz')
ax1.plot(S_nylon_FINAL[1,:],'xr--',label='Nylon, Tx: 2')#, f: '+str(f_medida[nfrec]/1e6)+' MHz')
ax1.plot(S_nylon_FINAL[2,:],'sg--',label='Nylon, Tx: 3')#, f: +str(f_medida[nfrec]/1e6)+' MHz')
ax1.plot(S_nylon_FINAL[3,:],'vy--',label='Nylon, Tx: 4')#, f: '+str(f_medida[nfrec]/1e6)+' MHz')
ax1.plot(S_nylon_FINAL[6,:],'^m--',label='Nylon, Tx: 7')#, f: '+str(f_medida[nfrec]/1e6)+' MHz')
ax1.set_ylim([-65, -25])
ax1.legend(loc="upper right")
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax1.set_xticklabels(['12', '13', '14', '15', '16', '17', '18']) 

plt.show()
