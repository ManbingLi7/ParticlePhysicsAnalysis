import numpy as np
from binnings_collection import *


def binning_energy():                                                                                                                                                                                     
    binning = np.array([0.338, 0.4185, 0.5077,0.6103,0.7264,0.8561,1.0045,1.1666,1.3476,1.5473,1.7659,2.0085,2.2753,2.5662,2.8812,                                                                         
                        3.2256,3.5996,4.0029,4.4413,4.9146,5.4229,5.9886,6.5553,7.1793,7.8608,8.5998,9.3963,10.2502,11.1616,12.1303])                                                                      
    return binning

binn = binning_energy()
print(get_bin_center(binn))
