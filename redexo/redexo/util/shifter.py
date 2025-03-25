#!/usr/bin/env python3

#Default Modules:
import numpy as np
import pandas as pd

#Other Modules:
from scipy.optimize import newton
from numpy import ndarray
from scipy import interpolate
##### Author: Alex Polanski #####


def spline_inter(x, y, xnew, k):
    """
        NB:
        -if k=1, we have a strict linear interpolation
        -scipy.interpolate.interp1d(x, y, kind='linear') equivalent as if if k=1
        -scipy.interpolate.interp1d(x, y, kind='cubic') slighly different (but near, see cases) as if if k=3

        if y is shifted like
        y[a+shift:b+shift]
        or xnew = x + shift
        or x = x - shift
        then: -shift>0 => ynew is blue shift
        -shift<0 => ynew is red shift
        NB: Be careful to the scale (not the same shift in x or y)

        keyword arguments:
        x -- Old x axis
        y -- Old y axis
        xnew -- New x axis
        k -- The Spline Order (1=linear, 3=cubic)

        """

    splflux = interpolate.InterpolatedUnivariateSpline(x, y, k=k)

    return splflux(xnew)

def calc_max_shift(velocity_list,pixel_size=1100.):
    nb_pixels = [rad_vel/pixel_size for rad_vel in list(velocity_list)]
    cutoff= int(round(abs(max(nb_pixels,key=abs))))
    return cutoff


def vel_rebin(vel_total,bin_set,data_pix,wave_set, cutoff,pixel_size=1.100):
    """
        Function that shifts all spectra by a velocity array and then realigns them
        INPUT: vel_total: array of the velocities (1D)
               bin_set: 1D array of the bins
               data_pix: 1D array of the spectral data, one axis wavelength bins
               wave_set: array of the wavelength axis
               cutoff: maximum number of bins that have to be discarded on the sides to not create empty bins
               pixel_size: optional. default value is the HARPS pixel size= 820 m/s
        OUTPUT: data_pix_shift: new array of the shifted data
                bin_set_shift: new bin array
                wave_set_shift: shifted wavelength array
        """
    pixel_shift = vel_total/pixel_size

    bin_set_shift=bin_set[cutoff:len(data_pix)-cutoff]
    wave_set_shift=wave_set[cutoff:len(data_pix)-cutoff]

    data_pix_shift=np.empty(len(bin_set_shift))

    data_pix_shift_interm = spline_inter(bin_set, data_pix, bin_set+pixel_shift, 1)

    data_pix_shift = data_pix_shift_interm[cutoff:len(data_pix_shift_interm)-cutoff]

    return data_pix_shift,  bin_set_shift, wave_set_shift








