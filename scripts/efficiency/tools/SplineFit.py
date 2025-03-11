import numpy as np


import numpy as np
from scipy.interpolate import UnivariateSpline

def cfSplineFit(x, p, FitOpt, sBoundary):
    if FitOpt & 1:  # Check if LogX bit is set
        xx = np.log(x[0])
    else:
        xx = x[0]
        
    nnode = int(p[0])
    node = np.zeros(nnode)
    ynode = np.zeros(nnode)
    
    for inode in range(nnode):
        if FitOpt & 2:  # Check if LogX bit is set
            node[inode] = np.log(p[inode + 1])
        else:
            node[inode] = p[inode + 1]
        
        if FitOpt & 4:  # Check if LogY bit is set
            ynode[inode] = np.log(p[inode + 1 + nnode])
        else:
            ynode[inode] = p[inode + 1 + nnode]
            
    sp3 = UnivariateSpline(node, ynode, s=sBoundary)
    
    value = sp3(xx)
    
    if FitOpt & 8 and xx > node[-1]:  # Check if ExtrapolateLE bit is set
        value = sp3(node[-1]) + sp3.derivative()(node[-1]) * (xx - node[-1])
    elif FitOpt & 16 and xx < node[0]:  # Check if ExtrapolateLB bit is set
        value = sp3(node[0]) + sp3.derivative()(node[0]) * (xx - node[0])
        
    if FitOpt & 32:  # Check if LogY bit is set
        value = np.exp(value)
        
    return value
