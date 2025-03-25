__all__ = ["gaussian_smooth","weight"]
import glob
from ..core import Dataset
from astropy.io import fits
import h5py
import numpy as np


def weight(w0,w,L):

    return(np.exp( -(w0-w)**2 / L**2) )

def gaussian_smooth(w,f,L):

    f_smooth = np.ones(len(f))

    for i in range(len(w)):
        fbar = np.sum(f* weight(w[i],w,L)) / np.sum(weight(w[i],w,L))

        f_smooth[i] = fbar

    return(f_smooth)

