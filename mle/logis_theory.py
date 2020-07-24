#
# Analytical estimation of the errors under
#
# gt : ReLU
# gs : logistic
#
from math import *
import sys
import numpy as np

def F(ltmp, ctmp):
    ztmp = ltmp - 1.0 + ctmp
    return sqrt(ztmp*ztmp + 4.0*ctmp) - ztmp

def calc_eapr(h,x):
    Dt0 = 0.5;
    Ds0 = 0.29338;
    Css0 = 0.25;
    Css1 = 0.04269;
    
    Cts0 = 1.0/(2.0*sqrt(2.0*pi));
    Cts1 = 0.1033

    ds = Ds0 - (Css0 + Css1)
    dts = Dt0 - (Cts0*Cts0/Css0) - (Cts1*Cts1/Css1)

    c0 = (ds + Css1)/Css0
    c1 = ds/Css1

    etmp0 = dts + (Cts0*Cts0/Css0)*(c0/(c0 + h))
    etmp1 = 0.5*(Cts1*Cts1/Css1)*F(h/float(x), c1)

    return etmp0 + etmp1    

def calc_egen_ml(h,x,n,st2):
    eapr = calc_eapr(h,x)
    return (eapr + st2)*n/float(n-h)
