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


fig8 = plt.figure(8)
ax2 = fig8.add_subplot(211)
ax2.semilogx(1e6*datos[:,0],datos[:,1],'b-',label='Incidente S12')
ax2.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,1],'b--',label='S12 Teflon')
ax2.semilogx(1e6*datos[:,0],datos[:,2],'r-',label='Incidente S13')
ax2.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,2],'r--',label='S13 Teflon')
ax2.semilogx(1e6*datos[:,0],datos[:,6],'g-',label='Incidente S17')
ax2.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,6],'g--',label='S17 Teflon')
ax2.semilogx(1e6*datos[:,0],datos[:,7],'m-',label='Incidente S18')
ax2.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,7],'m--',label='S18 Teflon')

ax2.set_ylim([-75, -25])
ax2.legend(loc="lower right")

ax3 = fig8.add_subplot(212)
ax3.semilogx(1e6*datos[:,0],datos[:,3],'b-',label='Incidente S14')
ax3.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,3],'b--',label='S14 Teflon')
ax3.semilogx(1e6*datos[:,0],datos[:,4],'r-',label='Incidente S15')
ax3.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,4],'r--',label='S15 Teflon')
ax3.semilogx(1e6*datos[:,0],datos[:,5],'g-',label='Incidente S16')
ax3.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,5],'g--',label='S16 Teflon')

ax3.set_ylim([-75, -25])
ax3.legend(loc="lower right")

#print(nylon.shape)
fig9 = plt.figure(9)
ax2 = fig9.add_subplot(211)
ax2.semilogx(1e6*datos[:,0],datos[:,6],'b-',label='Incidente S17')
ax2.semilogx(1e6*datos_cilindronylon[:,0],datos_cilindronylon[:,6],'b--',label='S17 Nylon')
ax2.semilogx(1e6*datos[:,0],datos[:,7],'r-',label='Incidente S18')
ax2.semilogx(1e6*datos_cilindronylon[:,0],datos_cilindronylon[:,7],'r--',label='S18 Nylon')
ax2.semilogx(1e6*datos[:,0],datos[:,4],'g-',label='Incidente S15')
ax2.semilogx(1e6*datos_cilindronylon[:,0],datos_cilindronylon[:,4],'g--',label='S15 Nylon')
ax2.semilogx(1e6*datos[:,0],datos[:,5],'m-',label='Incidente S16')
ax2.semilogx(1e6*datos_cilindronylon[:,0],datos_cilindronylon[:,5],'m--',label='S16 Nylon')

ax2.set_ylim([-75, -25])
ax2.legend(loc="lower right")

ax3 = fig9.add_subplot(212)
ax3.semilogx(1e6*datos[:,0],datos[:,1],'b-',label='Incidente S12')
ax3.semilogx(1e6*datos_cilindronylon[:,0],datos_cilindronylon[:,1],'b--',label='S12 Nylon')
ax3.semilogx(1e6*datos[:,0],datos[:,2],'r-',label='Incidente S13')
ax3.semilogx(1e6*datos_cilindronylon[:,0],datos_cilindronylon[:,2],'r--',label='S13 Nylon')
ax3.semilogx(1e6*datos[:,0],datos[:,3],'g-',label='Incidente S14')
ax3.semilogx(1e6*datos_cilindronylon[:,0],datos_cilindronylon[:,3],'g--',label='S14 Nylon')

ax3.set_ylim([-75, -25])
ax3.legend(loc="lower right")

### Diferencias
fig10 = plt.figure(10)
ax2 = fig10.add_subplot(211)
ax2.semilogx(1e6*datos[:,0],-datos[:,1]+datos_canonico[:,1],'b-',label='S12_teflon-S12inc')
ax2.semilogx(1e6*datos[:,0],-datos[:,2]+datos_canonico[:,2],'r-',label='S13_teflon-S13inc')
ax2.semilogx(1e6*datos[:,0],-datos[:,6]+datos_canonico[:,6],'m-',label='S17_teflon-S17inc')
ax2.semilogx(1e6*datos[:,0],-datos[:,7]+datos_canonico[:,7],'g-',label='S18_teflon-S18inc')
ax2.legend(loc="upper right")
ax2.grid(True)
#ax2.set_xlim([650e6, 900e6])

ax3 = fig10.add_subplot(212,sharex=ax2)
ax3.semilogx(1e6*datos[:,0],-datos[:,3]+datos_canonico[:,3],'y-',label='S14_teflon-S14inc')
ax3.semilogx(1e6*datos[:,0],-datos[:,4]+datos_canonico[:,4],'k-',label='S15_teflon-S15inc')
ax3.semilogx(1e6*datos[:,0],-datos[:,5]+datos_canonico[:,5],'xkcd:sky blue',label='S16_teflon-S16inc')
ax3.legend(loc="upper right")

ax3.grid(True)

#print(nylon.shape)

fig11 = plt.figure(11)
ax2 = fig11.add_subplot(211)
ax2.semilogx(1e6*datos[:,0],-datos[:,6]+datos_cilindronylon[:,6],'m-',label='S17_nylon-S17inc')
ax2.semilogx(1e6*datos[:,0],-datos[:,7]+datos_cilindronylon[:,7],'g-',label='S18_nylon-S18inc')
ax2.semilogx(1e6*datos[:,0],-datos[:,4]+datos_cilindronylon[:,4],'k-',label='S15_nylon-S15inc')
ax2.semilogx(1e6*datos[:,0],-datos[:,5]+datos_cilindronylon[:,5],'xkcd:sky blue',label='S16_nylon-S16inc')
ax2.legend(loc="upper right")
ax2.grid(True)
#ax2.set_xlim([650e6, 900e6])

ax3 = fig11.add_subplot(212,sharex=ax2)
ax3.semilogx(1e6*datos[:,0],-datos[:,1]+datos_cilindronylon[:,1],'b-',label='S12_nylon-S12inc')
ax3.semilogx(1e6*datos[:,0],-datos[:,2]+datos_cilindronylon[:,2],'r-',label='S13_nylon-S13inc')
ax3.semilogx(1e6*datos[:,0],-datos[:,3]+datos_cilindronylon[:,3],'y-',label='S14_nylon-S14inc')
ax3.legend(loc="upper right")

ax3.grid(True)

plt.show()
