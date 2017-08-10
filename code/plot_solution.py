from __future__ import division

import sys

import numpy as np

from pyFDM.utils import plot


    
#@profile
def main(csv):
    for c in csv:
        U = np.loadtxt(c, delimiter=",")
        plot(U)
        
        
    
if __name__ == "__main__":
    main(sys.argv[1:])