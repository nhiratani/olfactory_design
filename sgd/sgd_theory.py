#
# calculation of approximation error and estimation error under SGD
#
from math import *
import sys
import numpy as np
#from numpy import random as nrnd
#from numpy import linalg as nlg

Pi = np.pi

#ReLU coefficients
Css0 = 1.0/(2.0*pi)
Css1 = 1.0/4.0
Css2 = 1.0/(4.0*pi)
ds = 1.0/2.0 - (Css0 + Css1 + Css2)
    
co = (ds + Css1 + Css2)/Css0
c1 = (ds + Css2)/Css1
c2 = ds/Css2
    
def F(ltmp, ctmp):
    ztmp = ltmp - 1.0 + ctmp
    return sqrt(ztmp*ztmp + 4.0*ctmp) - ztmp

def calc_eapr(h,x):
    etmp0 = Css0*(1.0 - co/(co + h))
    etmp1 = Css1*(1.0 - 0.5*F(h/float(x), c1))
    if h > x:
        etmp2 = Css2*(1.0 - x/float(h))*(1.0 - 0.5*F( h/(0.5*x*x), c2/(1.0 - x/float(h)) ))
    else:
        etmp2 = 0.0
    return 1.0/2.0 - (etmp0 + etmp1 + etmp2)

#the rank of each component
def calc_Lrs(h,x):     
    L1 = min(x, h-1)
    L2 = max(0, min(x*(x-1)/2, h-x-1))
    Lr = max(0, h-x*(x+1)/2-1)
    return [1, L1, L2, Lr]

#the mean eigenvalues
def calc_lms(h,x):
    Lrs = calc_Lrs(h,x)
    L1 = Lrs[1]; L2 = Lrs[2]; Lr = Lrs[3]
    
    lm0 = Css0*(co + h)
    lm1 = Css1*(c1 + h/float(L1))
    if h > x+1:
        lm2 = Css2*(c2 + (1.0 - x/h)*(h-x)/float(L2))
    else:
        lm2 = ds #Css2*c2 
    lmr = ds
    lms = [lm0, lm1, lm2, lmr]

    return lms

#Calculation of the bundled errors
def calc_psis(h,x, st2, sr2):
    Lrs = calc_Lrs(h,x)
    lms = calc_lms(h,x)

    psi_zeros = []
    for i in range(4):
        psi_zeros.append( sr2*Lrs[i]*lms[i]/float(h) )

    psi_zeros[0] += Css0
    psi_zeros[1] += Css1*(1.0 - 0.5*F(h/x, c1))
    if h > x+1:
        psi_zeros[2] += Css2*(1.0 - x/h)*(1.0 - 0.5*F(h/(0.5*x*x), c2/(1.0 - x/h)))

    Lm = 0.5*h #the sum of all eigenvalues
    minf = (calc_eapr(h,x) + st2)/Lm
    psi_infs = []
    for i in range(4):
        psi_infs.append( Lrs[i]*lms[i]*minf )

    return psi_zeros, psi_infs, Lm

#sequence of estimation error 
def calc_eest_seq(h, x, N, st2, sr2):
    Lrs = calc_Lrs(h,x)
    lms = calc_lms(h,x)
    psi_zeros, psi_infs, Lm = calc_psis(h, x, st2, sr2)

    psi_tots = np.zeros((N))
    psis = np.zeros((4,N))
    for n in range(N):
        for i in range(4):
            if i == 0:
                psis[i][n] = psi_infs[i] + (psi_zeros[i] - psi_infs[i])*((1.0 - lms[i]/Lm)**n)
            else:
                psis[i][n] = psi_infs[i] + (psi_zeros[i] - psi_infs[i])*exp(-lms[i]*n/Lm)
        psi_tots[n] = psis[0][n] + psis[1][n] + psis[2][n] + psis[3][n]
    return psi_tots

#cumulative estimation error
def calc_eest_cm(h, x, N, st2, sr2):
    Lrs = calc_Lrs(h,x)
    lms = calc_lms(h,x)
    psi_zeros, psi_infs, Lm = calc_psis(h, x, st2, sr2)

    Psis = [0.0, 0.0, 0.0, 0.0]
    Psis[0] = psi_infs[0] + (psi_zeros[0] - psi_infs[0])*(1.0 - (1.0 - lms[0]/Lm)**N)/(N*lms[0]/Lm)
    Psis[1] = psi_infs[1] + (psi_zeros[1] - psi_infs[1])*(1.0 - exp(-N*lms[1]/Lm))/(N*lms[1]/Lm)
    Psis[2] = psi_infs[2]
    if h > x+1:
        Psis[2] += (psi_zeros[2] - psi_infs[2])*(1.0 - exp(-N*lms[2]/Lm))/(N*lms[2]/Lm)
    Psis[3] = psi_infs[3] + (psi_zeros[3] - psi_infs[3])*(1.0 - exp(-N*lms[3]/Lm))/(N*lms[3]/Lm)

    return Psis[0] + Psis[1] + Psis[2] + Psis[3]



