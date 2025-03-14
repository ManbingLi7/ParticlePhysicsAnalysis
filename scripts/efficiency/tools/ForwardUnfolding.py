import numpy as np
import 


def cfSplineFit(x, p, FitOpt, sBoundary):
    if FitOpt & 1:  # Check if LogX bit is set
        xx = np.log(x)
    else:
        xx = x
        
    nnode = int(p[0])
    xnode = np.zeros(nnode)
    ynode = np.zeros(nnode)
    
    for inode in range(nnode):
        if FitOpt & 2:  # Check if LogX bit is set
            xnode[inode] = np.log(p[inode + 1])
        else:
            xnode[inode] = p[inode + 1]
        
        if FitOpt & 4:  # Check if LogY bit is set
            ynode[inode] = np.log(p[inode + 1 + nnode])
        else:
            ynode[inode] = p[inode + 1 + nnode]
            
    sp3 = UnivariateSpline(xnode, ynode, s=sBoundary)
    
    value = sp3(xx)
    
    if FitOpt & 8 and xx > node[-1]:  # Check if ExtrapolateLE bit is set
        value = sp3(node[-1]) + sp3.derivative()(node[-1]) * (xx - node[-1])
    elif FitOpt & 16 and xx < node[0]:  # Check if ExtrapolateLB bit is set
        value = sp3(node[0]) + sp3.derivative()(node[0]) * (xx - node[0])
        
    if FitOpt & 32:  # Check if LogY bit is set
        value = np.exp(value)
        
    return value



def ChiSquareEvRate(self):
    dChiS = 0.0
    
    if verbose:
        print("Chis")
    
    if f1TrueFlux is None:
        print("ISS Event Rate not initialized")
    
    if verbose:
        for ipar in range(2 * nnode):
            print(f1TrueFlux.GetParameter(ipar + 1), end=" ")
        if FluxType > 0:
            print(f1TrueFlux.GetParameter(2 * nnode + 1), end=" ")
    
    if h1Rate is None:
        print("ISS Event Rate not initialized")
    
    for ibin in range(1, h1Rate.GetNbinsX() + 1):
        if h1Rate.GetBinLowEdge(ibin) < dLowerRigChis:
            continue
        if h1Rate.GetBinLowEdge(ibin) >= 100.0 and SPAN == 1:
            break
        
        if h1Rate.GetBinLowEdge(ibin) > dUpperRigChis:
            break
        
        dRateData = h1Rate.GetBinContent(ibin)
        dRateErrorData = h1Rate.GetBinError(ibin)
        
        dRateFUM = FUMCoreInvIntegration(h1Rate.GetBinLowEdge(ibin), h1Rate.GetBinLowEdge(ibin + 1))
        
        if dRateErrorData == 0:
            continue
        
        dChiS += ((dRateData - dRateFUM / h1Rate.GetBinWidth(ibin)) / dRateErrorData) ** 2
    
    return dChiS


def ChiSquareEvRateLogxLogy2(self, x):
    xx = np.exp(x[:2*nnode])
    
    for ipar in range(2 * nnode):
        self.f1TrueFlux.SetParameter(1 + ipar, xx[ipar])
        
    return self.ChiSquareEvRate()



def NumericalMinimizationLogxLogy(self, iMaxIter, dStep=0.01):
    minuit = Minuit.from_array_func(self.ChiSquareEvRateLogxLogy, np.zeros(2 * nnode + 1))
    minuit.errordef = 1  # This sets the error definition for the likelihood function
    minuit.strategy = 2  # Set the minimization strategy
    minuit.tol = 0.001  # Set the tolerance
    
    # Set the parameter limits and step sizes
    for ivar in range(2 * nnode + 1):
        minuit.limits[ivar] = (None, None)  # No parameter limits for now
        minuit.errors[ivar] = dStep
        minuit.step_size[ivar] = dStep

    start_time = time.time()
    print("Start minimizing with free log(x) & log(y) & const", time.ctime(start_time))
    
    # Perform the minimization
    minuit.migrad()
    
    end_time = time.time()
    print("Finish minimizing with free log(x) & log(y) & const", time.ctime(end_time))
    
    params = minuit.values
    
    print("Minimum:")
    for ivar in range(2 * nnode):
        print(np.exp(params[ivar]), end=", ")
    print(params[2 * nnode])
    
    chisq = self.ChiSquareEvRateLogxLogy(*params)
    print("chi square =", chisq)
    
    return chisq


