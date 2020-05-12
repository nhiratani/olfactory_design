#
# Three-layred feedforward model 
#
# Evaluation of the approximation, generalization, and test error
#
# gt : rectifier
# gs : rectifier
# Jo : Normalized Gaussian
# wo : Normalized Gaussian
# Jr : Normalized Gaussian
# wr : Maximum likelihood
#
from math import *
import sys
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg

Pi = 3.14159265
epsilon = 0.00000001

Lo = 500 #hidden unit number
Ntest = 30000 #sample size of the test data

#Rectifier
def calc_Prf(Ctmp, La, Lb):
    P1 = np.sqrt( np.ones((La,Lb)) - (1.0-epsilon)*np.multiply(Ctmp,Ctmp))
    P2 = np.multiply( Ctmp, np.arccos(-Ctmp*(1.0-epsilon)) )
    return P1 + P2

def calc_ers(Jo, wo, Jr, N, sigmat2):
    #Estimation of approximation error
    Lo = len(Jo); Lh = len(Jr); Lx = len(Jo[0])
    
    So = np.dot(Jo, np.transpose(Jo)); sqDo = np.sqrt( np.diag(So) )
    Co = np.divide(So, np.outer(sqDo,sqDo))
    Po = calc_Prf(Co, Lo, Lo)
    Go = np.multiply( np.outer(sqDo,sqDo), Po )/(2*Pi);
    sigmap1 = np.dot( wo, np.dot(Go, wo) )

    Sr = np.dot(Jr, np.transpose(Jr)); sqDr = np.sqrt( np.diag(Sr) )
    Cr = np.divide(Sr, np.outer(sqDr,sqDr)); Sr = None
    Pr = calc_Prf(Cr, Lh, Lh)
    Gr = np.multiply( np.outer(sqDr,sqDr), Pr )/(2*Pi); Pr = None;

    Sor = np.dot(Jo, np.transpose(Jr));
    Cor = np.divide(Sor, np.outer(sqDo,sqDr))
    Por = calc_Prf(Cor, Lo, Lh)
    Gor = np.multiply( np.outer(sqDo,sqDr), Por )/(2*Pi)    

    wr_opt = nlg.solve(Gr, np.dot(np.transpose(Gor),wo)); #Gr = None
    sigmap2 = np.dot(wo, np.dot(Gor, wr_opt))
    Cor = None; Por = None; #Gor = None;
    eapr = sigmap1 - sigmap2

    #Estimation of estimation, generalization, and test error
    xN = nrnd.normal(0.0, 1.0, (Lx, N))
    ytN = np.dot(wo, np.clip(np.dot(Jo, xN), 0, None)) + nrnd.normal(0,sqrt(sigmat2),(N))
    gsN = np.clip(np.dot(Jr, xN), 0, None)
    wrN = nlg.solve(np.dot(gsN, np.transpose(gsN)), np.dot(gsN, ytN))

    dwr = wrN - wr_opt
    eest = np.dot(dwr, np.dot(Gr, dwr))
    egen = sigmat2 + sigmap1 + np.dot(wrN, np.dot(Gr, wrN)) - 2.0*np.dot(wo, np.dot(Gor, wrN))

    xNtest = nrnd.normal(0.0, 1.0, (Lx, Ntest))
    ytNtest = np.dot(wo, np.clip(np.dot(Jo, xNtest), 0, None)) + nrnd.normal(0,sqrt(sigmat2),(Ntest))
    ysNtest = np.dot(wrN, np.clip(np.dot(Jr, xNtest), 0, None))
    etest = np.dot(ytNtest - ysNtest, ytNtest - ysNtest)/float(Ntest)  
    Gr = None; Gor = None
    
    return eapr, eest, egen, etest
    
def simul(Lx, N, sigmat2, ik):
    festr = 'data/ml_simul_Lx' + str(Lx) + '_N' + str(N) + '_st' + str(sigmat2) + '_ik' + str(ik) + '.txt'
    fwe = open(festr,'w')

    Lhs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 46, 50, 55, 60, 66, 72, 79, 86, 94, 103, 113, 124, 136, 149, 163, 179, 196, 215, 236, 259, 284, 312, 343, 377, 414, 455, 500, 550, 605, 665, 731, 804, 884, 972, 1069, 1175, 1292, 1421, 1563, 1719, 1890, 2079, 2286, 2514, 2765, 3041, 3345, 3679, 4046, 4450, 4895, 5384, 5922, 6514, 7165, 7881, 8669, 9535, 10488, 11536, 12689, 13957, 15352, 16887, 18575, 20432, 22475, 24722, 27194]#, 29913]#, 32904, 36194, 39813, 43794, 48173]

    Jo = nrnd.normal(0.0, 1.0/sqrt(float(Lx)), (Lo, Lx)) #normalized
    wo = nrnd.normal(0.0, 1.0/sqrt(float(Lo)), (Lo)) #zero-mean
    
    for lhidx in range(len(Lhs)):
        Lh = Lhs[lhidx]
        if Lh < N:
            Jr = nrnd.normal(0.0, 1.0/sqrt(float(Lx)), (Lh, Lx))
            etmps = calc_ers(Jo, wo, Jr, N, sigmat2)
            fwe.write( str(Lh) + " " + str(etmps[0]) + " " + str(etmps[1]) + " " + str(etmps[2]) + " " + str(etmps[3]) + "\n" )
            fwe.flush()

if __name__ == "__main__":
    param = sys.argv
    Lx = int(param[1])
    N = int(param[2])
    sigmat2 = float(param[3])
    ik = int(param[4])

    simul(Lx, N, sigmat2, ik)

