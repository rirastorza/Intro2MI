#!/usr/bin/env python3
"""
Plotea resultados variando M y niter en MoM

"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

#M = 80
#niter = 200 
#npzfile1 = np.load('test_M_'+str(M)+'_niter_'+str(niter)+'.npz')

M = 50
niter = 200 
npzfile2 = np.load('test_cil_M_'+str(M)+'_niter_'+str(niter)+'.npz')

M = 100
niter = 200 
npzfile3 = np.load('test_cil_M_'+str(M)+'_niter_'+str(niter)+'.npz')

M = 150
niter = 200 
npzfile4 = np.load('test_cil_M_'+str(M)+'_niter_'+str(niter)+'.npz')

#M = 250
#niter = 200 
#npzfile5 = np.load('test_M_'+str(M)+'_niter_'+str(niter)+'.npz')


parameters = {'axes.labelsize': 15,
          'axes.titlesize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'legend.fontsize': 15,
          'legend.loc': 'upper left'}
plt.rcParams.update(parameters)

nantena_inc = 3
fig, axs = plt.subplots(2, 1)
#pH y temp
color = 'tab:blue'
#axs[0].plot(abs(npzfile1['Es'][:,nantena_inc]),'-.',color = 'skyblue',label='M:80, niter:200') 
axs[0].plot(abs(npzfile2['Es'][:,nantena_inc]),'-',color = 'tab:blue',label='M:50, niter:200')
axs[0].plot(abs(npzfile3['Es'][:,nantena_inc]),'--',color = 'darkblue',label='M:100, niter:200')
axs[0].plot(abs(npzfile4['Es'][:,nantena_inc]),'o-',color = 'darkblue',label='M:150, niter:200')
#axs[0].plot(abs(npzfile5['Es'][:,nantena_inc]),'-',color = 'darkblue',label='M:250, niter:200')

axs[0].set_ylabel('abs(Es)', color=color)
#axs[0].set_xlabel('t [hr]')
#axs[0].tick_params(axis='y', labelcolor=color)
axs[0].legend(loc="upper right")


#axs[1].plot(np.angle(npzfile1['Es'][:,nantena_inc]),'-.',color = 'skyblue')
axs[1].plot(np.angle(npzfile2['Es'][:,nantena_inc]),'-',color = 'tab:blue')
axs[1].plot(np.angle(npzfile3['Es'][:,nantena_inc]),'--',color = 'darkblue')
axs[1].plot(np.angle(npzfile4['Es'][:,nantena_inc]),'o-',color = 'darkblue')
#axs[1].plot(np.angle(npzfile5['Es'][:,nantena_inc]),'-',color = 'darkblue')

axs[1].set_ylabel('angle(Es)', color=color)

plt.show()
