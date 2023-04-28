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
ax2 = fig8.add_subplot(241)
#ax2.semilogx(1e6*datos[:,0],datos[:,3],'b-',label='Incidente S14')
#ax2.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,3],'b--',label='S14 Teflon')
ax2.semilogx(1e6*datos[:,0],datos[:,4],'r-',label='Incidente S15')
ax2.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,4],'r--',label='S15 Teflon')
#ax2.semilogx(1e6*datos[:,0],datos[:,5],'g-',label='Incidente S16')
#ax2.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,5],'g--',label='S16 Teflon')

ax2.set_ylim([-75, -25])
ax2.legend(loc="lower left")
ax2.set_xlim([800e6, 1200e6])

ax3 = fig8.add_subplot(242)
#ax3.semilogx(1e6*datos[:,0],datos[:,10],'b-',label='Incidente S25')
#ax3.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,10],'b--',label='S25 Teflon')
ax3.semilogx(1e6*datos[:,0],datos[:,11],'r-',label='Incidente S26')
ax3.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,11],'r--',label='S26 Teflon')
#ax3.semilogx(1e6*datos[:,0],datos[:,12],'g-',label='Incidente S27')
#ax3.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,12],'g--',label='S27 Teflon')

ax3.set_ylim([-75, -25])
ax3.legend(loc="lower left")
ax3.set_xlim([800e6, 1200e6])

ax4 = fig8.add_subplot(243)
#ax4.semilogx(1e6*datos[:,0],datos[:,16],'b-',label='Incidente S36')
#ax4.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,16],'b--',label='S36 Teflon')
ax4.semilogx(1e6*datos[:,0],datos[:,17],'r-',label='Incidente S37')
ax4.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,17],'r--',label='S37 Teflon')
#ax4.semilogx(1e6*datos[:,0],datos[:,18],'g-',label='Incidente S38')
#ax4.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,18],'g--',label='S38 Teflon')

ax4.set_ylim([-75, -25])
ax4.legend(loc="lower left")
ax4.set_xlim([800e6, 1200e6])

ax5 = fig8.add_subplot(244)
#ax5.semilogx(1e6*datos[:,0],datos[:,21],'b-',label='Incidente S47')
#ax5.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,17],'b--',label='S47 Teflon')
ax5.semilogx(1e6*datos[:,0],datos[:,22],'r-',label='Incidente S48')
ax5.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,22],'r--',label='S48 Teflon')
#ax5.semilogx(1e6*datos[:,0],datos[:,3],'g-',label='Incidente S41')
#ax5.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,3],'g--',label='S41 Teflon')

ax5.set_ylim([-75, -25])
ax5.set_xlim([800e6, 1200e6])
ax5.legend(loc="lower left")

ax2 = fig8.add_subplot(245)
ax2.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,4]-datos[:,4],'r--',label='S15Teflon-S15inc')

ax2.set_ylim([-4, 4])
ax2.legend(loc="lower left")
ax2.set_xlim([800e6, 1200e6])

ax3 = fig8.add_subplot(246)
#ax3.semilogx(1e6*datos[:,0],datos[:,10],'b-',label='Incidente S25')
#ax3.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,10],'b--',label='S25 Teflon')
#ax3.semilogx(1e6*datos[:,0],datos[:,11],'r-',label='Incidente S26')
ax3.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,11]-datos[:,11],'r--',label='S26Teflon-S26inc')
#ax3.semilogx(1e6*datos[:,0],datos[:,12],'g-',label='Incidente S27')
#ax3.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,12],'g--',label='S27 Teflon')
ax3.set_ylim([-4, 4])
ax3.legend(loc="lower left")
ax3.set_xlim([800e6, 1200e6])

ax4 = fig8.add_subplot(247)
#ax4.semilogx(1e6*datos[:,0],datos[:,16],'b-',label='Incidente S36')
#ax4.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,16],'b--',label='S36 Teflon')
#ax4.semilogx(1e6*datos[:,0],datos[:,17],'r-',label='Incidente S37')
ax4.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,17]-datos[:,17],'r--',label='S37Teflon-S37inc')
#ax4.semilogx(1e6*datos[:,0],datos[:,18],'g-',label='Incidente S38')
#ax4.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,18],'g--',label='S38 Teflon')

ax4.set_ylim([-4, 4])
ax4.legend(loc="lower left")
ax4.set_xlim([800e6, 1200e6])

ax5 = fig8.add_subplot(248)
#ax5.semilogx(1e6*datos[:,0],datos[:,21],'b-',label='Incidente S47')
#ax5.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,17],'b--',label='S47 Teflon')
#ax5.semilogx(1e6*datos[:,0],datos[:,22],'r-',label='Incidente S48')
ax5.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,22]-datos[:,22],'r--',label='S48 Teflon')
#ax5.semilogx(1e6*datos[:,0],datos[:,3],'g-',label='Incidente S41')
#ax5.semilogx(1e6*datos_canonico[:,0],datos_canonico[:,3],'g--',label='S41 Teflon')

ax5.set_ylim([-4, 4])
ax5.set_xlim([800e6, 1200e6])
ax5.legend(loc="lower left")




plt.show()
