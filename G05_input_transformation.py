import numpy as np
def sig_fun(x,alp=0.1):
    return 1/(1+np.exp(-alp*x))

# inverse transform 
def sig_inv(y,alp=0.1):
    return -np.log((1-y)/y)/alp
