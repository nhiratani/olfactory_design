#
# Analytical estimation of the errors
#
from math import *
import sys
import numpy as np

def F(ltmp, ctmp):
    ztmp = ltmp - 1.0 + ctmp
    return sqrt(ztmp*ztmp + 4.0*ctmp) - ztmp

def calc_eapr(h,x):
    Css0 = 1.0/(2.0*pi)
    Css1 = 1.0/4.0
    Css2 = 1.0/(4.0*pi)
    ds = 1.0/2.0 - (Css0 + Css1 + Css2)
    
    co = (ds + Css1 + Css2)/Css0
    c1 = (ds + Css2)/Css1
    c2 = ds/Css2

    etmp0 = Css0*(1.0 - co/(co + h))
    etmp1 = Css1*(1.0 - 0.5*F(h/float(x), c1))
    if h > x:
        etmp2 = Css2*(1.0 - x/float(h))*(1.0 - 0.5*F( h/(0.5*x*x), c2/(1.0 - x/float(h)) ))
    else:
        etmp2 = 0.0
    return 1.0/2.0 - (etmp0 + etmp1 + etmp2)

def calc_egen_ml(h,x,n,st2):
    eapr = calc_eapr(h,x)
    return (eapr + st2)*n/float(n-h)
