#
# Three-layred feedforward model of invertebrate olfaction
#
# (Evolutional) learning of full connections
#
# Mini-batch on Jp using the optimal wp[Jp]
#
# Low-precision by adding noise
#
# gt : rectifier
# gs : rectifier
# Jo : normarlized Gaussian
# wo : normarlized Gaussian
#
#import os
#os.environ["MKL_NUM_THREADS"] = '4'
#os.environ["NUMEXPR_NUM_THREADS"] = '4'
#os.environ["OMP_NUM_THREADS"] = '4'
from math import *
import sys
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg

Pi = 3.14159265
epsilon = 0.00000001

Lo = 500
Tmax = 100000
B = Lo

#Rectifier
def calc_Prf(Ctmp, La, Lb):
    P1 = np.sqrt( np.ones((La,Lb)) - (1.0-epsilon)*np.multiply(Ctmp,Ctmp))
    P2 = np.multiply( Ctmp, np.arccos(-Ctmp*(1.0-epsilon)) )
    return P1 + P2

def calc_sigmap1(Jo, wo):
    Lh = len(Jo)
    So = np.dot(Jo, np.transpose(Jo)); sqDo = np.sqrt( np.diag(So) )
    Co = np.divide(So, np.outer(sqDo,sqDo))
    Po = calc_Prf(Co, Lh, Lh)
    Go = np.multiply( np.outer(sqDo,sqDo), Po )/(2*Pi);
    sigmap1 = np.dot( wo, np.dot(Go, wo) )
    return sigmap1

def calc_er(Jo, wo, Jp, sigmap1):
    Lh = len(Jo); Lp = len(Jp)
    So = np.dot(Jo, np.transpose(Jo)); sqDo = np.sqrt( np.diag(So) )

    Sp = np.dot(Jp, np.transpose(Jp)); sqDp = np.sqrt( np.diag(Sp) )
    Sop = np.dot(Jo, np.transpose(Jp));
    Cp = np.divide(Sp, np.outer(sqDp,sqDp)); Sp = None
    Pp = calc_Prf(Cp, Lp, Lp)
    Gp = np.multiply( np.outer(sqDp,sqDp), Pp )/(2*Pi);
    Pp = None; 
    
    Cop = np.divide(Sop, np.outer(sqDo,sqDp))
    Pop = calc_Prf(Cop, Lh, Lp)
    Gop = np.multiply( np.outer(sqDo,sqDp), Pop )/(2*Pi)
    Gpo = np.transpose(Gop)

    GpinvGpowo = nlg.solve(Gp, np.dot(Gpo,wo)); Gp = None
    sigmap2 = np.dot(wo, np.dot(Gop, GpinvGpowo))
    Cop = None; Pop = None; Gop = None; Gpo = None; Gpo_zero = None; 
    return sigmap1-sigmap2, GpinvGpowo

def calc_dJp(Jo, wo, Jp, wp):
    Lp = len(wp)
    xB = nrnd.normal(0.0, 1.0, (Lx, B))
    ytB = np.dot( wo, np.clip( np.dot(Jo, xB), 0.0, None) )
    ysB = np.dot( wp, np.clip( np.dot(Jp, xB), 0.0, None) )
    wpdgp = np.multiply(np.outer( wp, np.ones((B)) ), 0.5*( np.sign(np.dot(Jp, xB)) + np.ones((Lp,B)) ))
    
    return np.dot(wpdgp ,np.dot(np.diag(ytB-ysB), np.transpose(xB)) )/float(B)

#Low bit encoding by adding noise
def calc_JwpK_noise(Jp, wp, sbit):
    Lx = len(Jp[0]); Lp = len(Jp)
    gm = sqrt(1.0 - exp(-2.0*log(2.0)*sbit))
    sigw = np.std(wp); wnoise = sqrt(1.0 - gm*gm)*sigw
    sigJ = np.std(Jp); Jnoise = sqrt(1.0 - gm*gm)*sigJ

    JpK = gm*Jp + nrnd.normal(0.0, Jnoise, (Lp,Lx))
    wpK = gm*wp + nrnd.normal(0.0, wnoise, (Lp))
    return JpK, wpK

#Low bit encoding by discretization
def calc_JwpK_disc(Jp, wp, sbit):
    sbit = int(sbit)
    Kp = 2**sbit
    Lx = len(Jp[0]); Lp = len(Jp)
    
    JpK = np.zeros((Lp,Lx))
    Jpmin = np.min(Jp); Jpmax = np.max(Jp);
    dJp = (Jpmax - Jpmin)/float(Kp)
    for i in range(Lp):
        for j in range(Lx):
            JpK[i][j] = Jpmin + (floor((Jp[i][j]-Jpmin)/dJp) + 0.5)*dJp

    wpK = np.zeros((Lp))
    wpmin = np.min(wp); wpmax = np.max(wp);
    dwp = (wpmax - wpmin)/float(Kp)
    for i in range(Lp):
        wpK[i] = wpmin + (floor((wp[i]-wpmin)/dwp) + 0.5)*dwp
        return JpK, wpK

def calc_test_er(Jo, wo, Jp, wp, Jr, N, sigmat2):
    Lh = len(Jr); Lp = len(wp)

    wr_hat = np.zeros((Lh))
    ertmp = 0.0
    
    for n in range(N):
        xn = nrnd.normal(0.0, 1.0, (Lx))
        ytn = np.dot(wo, np.clip(np.dot(Jo, xn), 0, None))
        if sigmat2 > 0.0:
            ytn += nrnd.normal(0,sqrt(sigmat2))
        ypn = np.dot(wp, np.clip(np.dot(Jp, xn), 0, None))

        gsn = np.clip(np.dot(Jr, xn), 0.0, None)
        ysn = np.dot(wr_hat, gsn)
        dyn = ytn - (ypn + ysn)

        ertmp = ertmp + (1.0/float(N))*np.multiply(dyn,dyn)
        eta_zero = 2.0/float( Lh + np.clip(n-Lh,0,None) )
        wr_hat += (eta_zero*dyn)*gsn
    return ertmp

def simul(Lx, N, eta_dJ, sigmat2, G, sbit, ik):
    festr = 'data/dual_simul_e_Lx' + str(Lx) + '_N' + str(N) + '_ej' + str(eta_dJ) + '_st' + str(sigmat2) + '_g' + str(G) + '_sb' + str(sbit) + '_ik' + str(ik) + '.txt'
    fwe = open(festr,'w')
    fpstr = 'data/dual_simul_p_Lx' + str(Lx) + '_N' + str(N) + '_ej' + str(eta_dJ) + '_st' + str(sigmat2) + '_g' + str(G) + '_sb' + str(sbit) + '_ik' + str(ik) + '.txt'
    fwp = open(fpstr,'w')

    Lhs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 46, 50, 55, 60, 66, 72, 79, 86, 94, 103, 113, 124, 136, 149, 163, 179, 196, 215, 236, 259, 284, 312, 343, 377, 414, 455, 500, 550, 605, 665, 731, 804, 884, 972, 1069, 1175, 1292, 1421, 1563, 1719, 1890, 2079, 2286, 2514, 2765, 3041, 3345, 3679, 4046, 4450, 4895, 5384, 5922, 6514, 7165, 7881, 8669, 9535, 10488, 11536, 12689, 13957, 15352, 16887, 18575, 20432, 22475, 24722, 27194, 29913, 32904, 36194, 39813, 43794, 48173, 52990, 58289, 64117, 70528, 77580, 85338, 93871, 103258]

    Jo = nrnd.normal(0.0, 1.0/sqrt(float(Lx)), (Lo, Lx)) #normalized
    wo = nrnd.normal(0.0, 1.0/sqrt(float(Lo)), (Lo)) #zero-mean
    sigmap1 = calc_sigmap1(Jo, wo)
    
    Lp = int( np.round( G/(sbit*(Lx+1.0)) ) ) #round, not floor
    Jp = nrnd.normal(0, 1.0, (Lp, Lx))/sqrt(float(Lx))
    wp = np.zeros((Lp)); wptmp = np.zeros((Lp))

    er_new, wp = calc_er(Jo, wo, Jp, sigmap1); er_old = 0.0
    fwetmp = ""

    #Evolutionary learning
    for t in range(Tmax):
        er_old = er_new
        dJ = eta_dJ*er_new
        Jptmp = Jp + dJ*calc_dJp(Jo, wo, Jp, wp)
        er_new, wptmp = calc_er(Jo, wo, Jptmp, sigmap1)
        
        Jp = Jptmp; wp = wptmp
        if t%100 == 0:
            fwp.write( str(t) + " " + str(er_new) + "\n" );
            fwp.flush()
    
    #Developmental learning
    JpK, wpK = calc_JwpK_noise(Jp, wp, sbit)
    for lhidx in range(len(Lhs)):
        Lh = Lhs[lhidx]
        Jr = nrnd.normal(0.0, 1.0/sqrt(float(Lx)), (Lh, Lx))
        etmp = calc_test_er(Jo, wo, JpK, wpK, Jr, N, sigmat2)
        fwe.write( str(sbit) + " " + str(Lh) + " " + str(etmp) + "\n" ); fwe.flush()

if __name__ == "__main__":
    param = sys.argv
    Lx = int(param[1]) #Input layer size
    N = int(param[2]) #The number of training samples
    eta_dJ = float(param[3]) #Learning rate of Jp learning
    sigmat2 = float(param[4]) #teacher noise
    G = float(param[5]) #Genetic budget
    sbit = float(param[6]) #Bit per synapse
    ik = int(param[7]) #simulation id
    
    simul(Lx, N, eta_dJ, sigmat2, G, sbit, ik)

