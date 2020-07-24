#
# Readout of the error curves as function of the hidden layer size Lh
#
from math import *
import sys
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg
from ml_theory import calc_eapr, calc_egen_ml
#from scipy import integrate as scint
#from scipy import special as scisp
#from scipy import stats as scist

import matplotlib.pyplot as plt
from pylab import cm


xmax = 100000
n = 30000
st2 = 0.1

dhstep = 1.01
dxstep = 1.05

x = 50#100 #[30,45,67,100,150]
hs = [8]
while hs[-1] < n:
    hs.append( int(float(hs[-1]*dhstep))+1 )
hs = hs[:-1]
hlen = len(hs)

eaprs = np.zeros((hlen))
eests = np.zeros((hlen))
egens = np.zeros((hlen))
for hidx in range(hlen):
    h = hs[hidx]
    eaprs[hidx] = calc_eapr(h,x)
    eests[hidx] = (st2 + eaprs[hidx])*h/float(n-h)
    egens[hidx] = calc_egen_ml(hs[hidx],x,n,st2)

svfg = plt.figure()
fig1 = plt.subplot()
plt.plot(hs, eaprs, 'b', lw=1.0);
plt.plot(hs, eests, 'g', lw=1.0);
plt.plot(hs, egens, 'r', lw=1.0);
plt.axvline( hs[np.argmin(egens)], color='k' )

#Readout from simulation
Lx = x
N = n
sigmat2 = st2#1.0#0.1
iks = range(10)#
ikmax = len(iks)

Lhs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 46, 50, 55, 60, 66, 72, 79, 86, 94, 103, 113, 124, 136, 149, 163, 179, 196, 215, 236, 259, 284, 312, 343, 377, 414, 455, 500, 550, 605, 665, 731, 804, 884, 972, 1069, 1175, 1292, 1421, 1563, 1719, 1890, 2079, 2286, 2514, 2765, 3041, 3345, 3679, 4046, 4450, 4895, 5384, 5922, 6514, 7165, 7881, 8669, 9535, 10488, 11536, 12689, 13957, 15352, 16887, 18575, 20432, 22475, 24722, 27194]#, 29913]#, 32904, 36194, 39813, 43794, 48173]
lhmax = len(Lhs)

#err = (eapr, eest, egen, etest)
errs = np.zeros( (4,lhmax) )
err_sigs = np.zeros( (4,lhmax) )
for ik in iks:
    lidx = 0; 
    festr = 'data/ml_simul_Lx' + str(Lx) + '_N' + str(N) + '_st' + str(sigmat2) + '_ik' + str(ik) + '.txt'
    for line in open(festr, 'r'):
        ertmps = line[:-1].split(" ")
        for qidx in range(4):
            ertmp = float(ertmps[1+qidx])
            errs[qidx][lidx] += ertmp/float(ikmax)
            err_sigs[qidx][lidx] += ertmp*ertmp/float(ikmax)
        lidx += 1
    print ik, lidx

for qidx in range(4):
    err_sigs[qidx] = np.sqrt( err_sigs[qidx] - np.multiply(errs[qidx], errs[qidx]) )
    
sp_Lhs = []; sp_errs = []; sp_err_sigs = []
for qidx in range(4):
    sp_errs.append([]); sp_err_sigs.append([])
for hidx in range(0,lhmax,5):
    sp_Lhs.append( Lhs[hidx] );
    for qidx in range(4):
        sp_errs[qidx].append( errs[qidx][hidx] )
        sp_err_sigs[qidx].append( err_sigs[qidx][hidx] )
    
plt.errorbar(sp_Lhs, sp_errs[0], sp_err_sigs[0], marker='', c='b', lw=0, elinewidth=2.5, mew=1.5, capsize=3.0)
plt.errorbar(sp_Lhs, sp_errs[1], sp_err_sigs[1], marker='', c='g', lw=0, elinewidth=2.5, mew=1.5, capsize=3.0)
plt.errorbar(sp_Lhs, sp_errs[3], sp_err_sigs[3], marker='', c='r', lw=0, elinewidth=2.5, mew=1.5, capsize=3.0)
plt.axvline( Lhs[np.argmin(errs[3])], color='k', ls='--')

plt.loglog(subsx=[-1],subsy=[-1])
plt.xlim(9,30000)
plt.ylim(0.0001,10.0)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
svfg.savefig('fig_ml_simul_error_h_curve1_Lx' + str(Lx) + '_N' + str(N) + '_st' + str(sigmat2) + '.pdf')

