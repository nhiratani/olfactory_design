#
# Three-layred feedforward model of invertebrate olfaction
#
# A model with single output unit
#
# Evaluation of the generalization error under SGD learning
#
# gt : rectifier
# gs : rectifier
# Jo : Normarlized Gaussian
# wo : Normalized Gaussian
# Jr : Normalized Gaussian
#
import os
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
from math import *
import sys
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg
#from scipy import integrate as scint
#from scipy import special as scisp
#from scipy import stats as scist 
#import matplotlib.pyplot as plt
#from pylab import cm

Pi = 3.14159265
epsilon = 0.00000001

Lo = 500 #hidden unit number

def calc_ers(Jo, wo, Jr, N, sigmat2, sigmaR2):
    Lh = len(Jr)
    ertmp = 0.0
    eta = 2.0/float(Lh)
    if sigmaR2 > 0.0:
        wr_hat = nrnd.normal(0.0, sqrt(sigmaR2/float(Lh)), (Lh))#
    else:
        wr_hat = np.zeros((Lh))

    for n in range(N):
        xn = nrnd.normal(0.0, 1.0, (Lx))
        ytn = np.dot(wo, np.clip(np.dot(Jo, xn), 0, None))
        if sigmat2 > 0.0:
            ytn += nrnd.normal(0,sqrt(sigmat2))
        gsn = np.clip(np.dot(Jr, xn), 0.0, None)
        ysn = np.dot(wr_hat, gsn)

        ertmp += (ytn - ysn)*(ytn - ysn)
        wr_hat += eta*(ytn - ysn)*gsn
    return ertmp/float(N)
    
def simul(Lx, N, sigmat2, sigmaR2, ik):
    festr = 'data/sgd_simul_Lx' + str(Lx) + '_N' + str(N) + '_st' + str(sigmat2) + '_sr' + str(sigmaR2) + '_ik' + str(ik) + '.txt'
    fwe = open(festr,'w')

    Lhs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 46, 50, 55, 60, 66, 72, 79, 86, 94, 103, 113, 124, 136, 149, 163, 179, 196, 215, 236, 259, 284, 312, 343, 377, 414, 455, 500, 550, 605, 665, 731, 804, 884, 972, 1069, 1175, 1292, 1421, 1563, 1719, 1890, 2079, 2286, 2514, 2765, 3041, 3345, 3679, 4046, 4450, 4895, 5384, 5922, 6514, 7165, 7881, 8669, 9535, 10488, 11536, 12689, 13957, 15352, 16887, 18575, 20432, 22475, 24722, 27194, 29913]#, 32904, 36194, 39813, 43794, 48173, 52990, 58289, 64117, 70528, 77580, 85338, 93871, 103258]
    
    Jo = nrnd.normal(0.0, 1.0/sqrt(float(Lx)), (Lo, Lx)) 
    wo = nrnd.normal(0.0, 1.0/sqrt(float(Lo)), (Lo))

    for lhidx in range(len(Lhs)):
        Lh = Lhs[lhidx]
        
        Jr = nrnd.normal(0.0, 1.0/sqrt(float(Lx)), (Lh, Lx))
        etmp = calc_ers(Jo, wo, Jr, N, sigmat2, sigmaR2)
        fwe.write( str(Lh) + " " + str(etmp) + "\n" )
        fwe.flush()

if __name__ == "__main__":
    param = sys.argv
    Lx = int(param[1])
    N = int(param[2])
    sigmat2 = float(param[3])
    sigmaR2 = float(param[4])
    ik = int(param[5])

    simul(Lx, N, sigmat2, sigmaR2, ik)

