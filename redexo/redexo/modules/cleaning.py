__all__ = ['FillNaNsModule','OutlierFlaggingModule', 'PolynomialContinuumRemovalModule', 'FlagAbsorptionEmissionModule','GaussianContinuumRemovalModule','SavGolContinuumRemovalModule','WavelengthCutModule','SimpleNormalizationModule','ScipyGaussianContinuumRemovalModule','ShiftStellarRestFrameModule','ShiftStellarRestFrameModule2','RemoveBlazeModule','SigmaClipInterpolateModule','HotPixelRemovalModule','SubtractMasterTemplateModule','RemoveHighDeviationPixelsModule','DivideVarianceModule']
from .base import Module
import numpy as np
import matplotlib.pyplot as plt
from ..util import smoothing,shifter
import scipy
from astropy import constants as const

class FlagAbsorptionEmissionModule(Module):
    '''
    Flags parts of the spectra that have less flux than flux_lower_limit or more flux than flux_upper_limit
    '''
    def initialise(self, flux_lower_limit=0, flux_upper_limit=np.inf, relative_to_continuum=False):
        self.flux_lower_limit = flux_lower_limit
        self.flux_upper_limit = flux_upper_limit
        self.relative_to_continuum = relative_to_continuum

    def process(self, dataset, debug=False):
        if self.relative_to_continuum:
            continuum_removed_data = PolynomialContinuumRemovalModule(poly_order=3)(dataset.copy())
            spec_norm = np.nanmedian(continuum_removed_data.spec, axis=0)
        else:
            spec_norm = np.nanmedian(dataset.spec, axis=0)
        mask = (spec_norm>self.flux_lower_limit)*(spec_norm<self.flux_upper_limit)
        if debug:
            print('Masking {0:.2f}% of the data'.format(np.sum(~mask)/mask.size*100))
        dataset.spec[:,~mask] = np.nan
        return dataset

class OutlierFlaggingModule(Module):
    def initialise(self, sigma=5):
        self.sigma = sigma

    def process(self, dataset, debug=False):
        std = np.nanstd(dataset.spec)
        mean = np.nanmean(dataset.spec)
        outliers = (np.abs(dataset.spec - mean)/std)>self.sigma
        if debug:
            print('Masking {0:.2f}% of the data'.format(np.sum(outliers)/outliers.size*100))
        dataset.spec[outliers] = np.nan
        return dataset


class PolynomialContinuumRemovalModule(Module):
    def initialise(self, poly_order=3):
        self.poly_order = poly_order

    def process(self, dataset, debug=False):
        for exp in range(dataset.num_exposures):
            nans = np.isnan(dataset.spec[exp])
            cont_model = np.poly1d(np.polyfit(dataset.wavelengths[exp][~nans], dataset.spec[exp][~nans], self.poly_order))
            continuum = cont_model(dataset.wavelengths[exp][~nans])
            dataset.spec[exp][~nans] = dataset.spec[exp][~nans]/continuum
            dataset.errors[exp][~nans] = dataset.errors[exp][~nans]/continuum
        return dataset

class GaussianContinuumRemovalModule(Module):
    def initialise(self, L=5):
        self.L = L
    def process(self, dataset, debug=False):
        for exp in range(dataset.num_exposures):
            nans = np.isnan(dataset.spec[exp])
            cont_model = smoothing.gaussian_smooth(dataset.wavelengths[exp][~nans],dataset.spec[exp][~nans],self.L)
            dataset.spec[exp][~nans] = dataset.spec[exp][~nans]/cont_model
            dataset.errors[exp][~nans] = dataset.errors[exp][~nans]/cont_model
        return dataset

class ScipyGaussianContinuumRemovalModule(Module):
    def initialise(self, sigma=5):
        self.sigma = sigma
    def process(self, dataset, debug=False):
        for exp in range(dataset.num_exposures):
            nans = np.isnan(dataset.spec[exp])
            cont_model = scipy.ndimage.gaussian_filter1d(dataset.spec[exp][~nans],self.sigma,truncate=6)
            dataset.spec[exp][~nans] = dataset.spec[exp][~nans] - cont_model
            #dataset.errors[exp][~nans] = dataset.errors[exp][~nans]/cont_model
        return dataset


class SavGolContinuumRemovalModule(Module):
    def initialise(self, window=501,polyorder=3):
        self.window = window
        self.polyorder = polyorder
    def process(self, dataset, debug=False):
        for exp in range(dataset.num_exposures):
            nans = np.isnan(dataset.spec[exp])
            cont_model = scipy.signal.savgol_filter(dataset.spec[exp][~nans],window_length=self.window,polyorder=self.polyorder)
            dataset.spec[exp][~nans] = dataset.spec[exp][~nans]/cont_model
            dataset.errors[exp][~nans] = dataset.errors[exp][~nans]/cont_model
        return dataset

class SimpleNormalizationModule(Module):
    #Simple normalization module. Dvides each spectrum its mean.
    def initialise(self,div_value=None):
        self.div_value=div_value

    def process(self,dataset,debug=False):
        normed = ((dataset.spec[:,0,:].T / np.mean(dataset.spec[:,0,:],axis=1)).T)
        dataset.spec = normed[:,np.newaxis,:]
        return dataset

class DivideVarianceModule(Module):
    #Simple normalization module. Dvides each spectrum its variance.
    def initialise(self,div_value=None):
        self.div_value=div_value

    def process(self,dataset,debug=False):
        normed = ((dataset.spec[:,0,:].T / np.std(dataset.spec[:,0,:],axis=1)**2).T)
        dataset.spec = normed[:,np.newaxis,:]
        return dataset

class WavelengthCutModule(Module):
    def initialise(self, low, high):
        self.low = low
        self.high = high
    def process(self, dataset, debug=False):
        if self.low is None:
            idx_low = 0
        else:
            a = np.abs(dataset.wavelengths[0,0,:]-self.low)
            idx_low = np.where((a == np.min(a)))[0][0] #the worst code on the planet

        if self.high is None:
            idx_high = len(dataset.wavelengths[0,0,:])-1
        else:
            a = np.abs(dataset.wavelengths[0,0,:]-self.high)
            idx_high = np.where((a == np.min(a)))[0][0]

        
        dataset.wavelengths = dataset.wavelengths[:,:,idx_low:idx_high]
        dataset.spec = dataset.spec[:,:,idx_low:idx_high]
        dataset.errors = dataset.errors[:,:,idx_low:idx_high]
        return dataset 

class RemoveHighDeviationPixelsModule(Module):
    def initialise(self,cut_off=2):
        self.cut_off = cut_off
    def process(self,dataset,debug=False):
        stds = np.nanstd(dataset.spec[:,0,:],axis=0)
        idx = np.where( stds > np.percentile(stds, (100-self.cut_off)))[0]
        dataset.spec[:,0,idx] = np.nan

        return(dataset)


class ShiftStellarRestFrameModule(Module):
    def initialise(self, target, ks, vsys):
        self.target = target
        self.kp = target.Kp
        self.per = target.orbital_period
        self.t0 = target.T0
        self.vsys = vsys
        self.ks = ks
    def process(self, dataset, debug=False):
        phases = self.target.orbital_phase(dataset.obstimes)
        vbar = dataset.vbar*1000.
        vel_star = self.ks*np.sin(2*np.pi*phases)
        vel_stellar_shift = -vbar+self.vsys-vel_star
        
        cutoff = shifter.calc_max_shift(vel_stellar_shift)
        bin_obs = np.arange(len(dataset.spec[0,0,:]))
        data_SRF, bin_shift, wave_SRF_keep = shifter.vel_rebin(vel_stellar_shift[0] ,bin_obs ,dataset.spec[0,0,:], dataset.wavelengths[0,0,:], cutoff)
        magic_number=len(data_SRF)
        bin_obs = np.arange(len(dataset.spec[0,0,:]))
        data_SRF = np.zeros(shape=(len(dataset.spec[:,0,:]),magic_number))
        wave_SRF = np.zeros(shape=(len(dataset.spec[:,0,:]),magic_number))
        
        for k in range(len(dataset.spec[:,0,:])):
            data_SRF[k,:], bin_shift, wave_SRF[k,:] = shifter.vel_rebin(vel_stellar_shift[k] ,bin_obs ,dataset.spec[k,0,:], dataset.wavelengths[k,0,:], cutoff)
            #wave_SRF[k,:] = wave_SRF_keep #this gives every exposure the EXACT same wavelength grid

        dataset.spec, dataset.wavelengths = data_SRF[:,np.newaxis,:], wave_SRF[:,np.newaxis,:]
        dataset.errors = np.ones_like(dataset.spec)
        return dataset

class ShiftStellarRestFrameModule2(Module):
    def initialise(self, target, ks, vsys, correct_vbary=True):
        self.target = target
        self.kp = target.Kp
        self.per = target.orbital_period
        self.t0 = target.T0
        self.vsys = vsys
        self.ks = ks
        self.correct_vbary = correct_vbary
    def process(self, dataset, debug=False):
        phases = self.target.orbital_phase(dataset.obstimes)
        vbar = dataset.vbar
        vel_star = self.ks*np.sin(2*np.pi*phases)

        if not self.correct_vbary:
            vel_stellar_shift = self.vsys+vel_star

        else:
            vel_stellar_shift = -vbar+self.vsys-vel_star
            #vel_stellar_shift = -vbar-self.vsys+vel_star #Seidels notebook    


        cutoff = shifter.calc_max_shift(vel_stellar_shift)
        wave_grid = np.linspace(dataset.wavelengths.min(),dataset.wavelengths.max(),len(dataset.wavelengths[0,0,:]))

        wave_grid = wave_grid[cutoff:len(wave_grid)-cutoff]
        data_SRF = np.zeros(dataset.spec[:,0,0:len(wave_grid)].shape)
        wave_SRF = np.zeros(dataset.spec[:,0,0:len(wave_grid)].shape)


        for exp in range(dataset.num_exposures):
            beta = 1-vel_stellar_shift[exp]/const.c.to('km/s').value
            new_wl = beta*dataset.wavelengths[exp]
            new_spec = shifter.spline_inter(new_wl,dataset.spec[exp],wave_grid,1)
            data_SRF[exp,:] = new_spec
            wave_SRF[exp,:] = wave_grid
        
        dataset.spec, dataset.wavelengths = data_SRF[:,np.newaxis,:], wave_SRF[:,np.newaxis,:]
        dataset.errors = np.ones_like(dataset.spec)
        
        return dataset

class RemoveBlazeModule(Module):
    #aurora's updated version of blaze removal 
    def initialise(self,sigma):
        self.sigma = sigma
    def process(self, dataset, debug=False):
        filtered_spec = scipy.ndimage.gaussian_filter1d(np.nanmean(dataset.spec[:,0,:],axis=0), 9)

        div = dataset.spec[:,0,:] / filtered_spec

        blaze = scipy.ndimage.gaussian_filter1d(div,self.sigma,axis=1)

        new_spec = dataset.spec[:,0,:]/blaze

        dataset.spec = new_spec[:,np.newaxis,:]

        
        return(dataset)

class SigmaClipInterpolateModule(Module):
    #Function to find wavelength pixels with flux values more than some sigma away from mean
    #Best used after common blaze removal
    
    def initialise(self,sigma):
        self.sigma = sigma
    def process(self, dataset, debug=False):
        means = np.nanmean(dataset.spec[:,0,:],axis=0) #get the mean value of each wavlength pixel
        stds = np.nanstd(dataset.spec[:,0,:],axis=0) #get the standard deviation of each wavelength pixel
        for exp in range(dataset.num_exposures):
            sigs = np.abs(dataset.spec[exp,0,:]-means) / stds #number of sigma each flux value is away from the overall mean per wavlength bin
            idx=np.where(sigs<self.sigma)[0] #get the indices of the flux values that are BELOW the specified sigma
            # print(f"wavelengths shape: {np.shape(dataset.wavelengths)}\tspec shape: {np.shape(dataset.spec)}")
            corrected_flux = scipy.interpolate.interp1d(dataset.wavelengths[exp,0,:][idx],dataset.spec[exp,0,:][idx],'linear',fill_value='extrapolate')(dataset.wavelengths[exp,0,:])
            dataset.spec[exp,:,:] = corrected_flux[np.newaxis,np.newaxis,:]

        return dataset

class FillNaNsModule(Module):
    def initialise(self,hold=False):
        self.hold = hold

    def process(self,dataset, debug=False):

        for exp in range(dataset.num_exposures):
            nan_idx = np.where(np.isnan(dataset.spec[exp,0,:]))[0] 
            good = np.where(~np.isnan(dataset.spec[exp,0,:]))[0]
            if len(nan_idx!=0):

                new_spec = scipy.interpolate.interp1d(dataset.wavelengths[exp,0,:][good], dataset.spec[exp,0,:][good],fill_value='extrapolate')(dataset.wavelengths[exp,0,:])

                dataset.spec[exp,:,:] = new_spec[np.newaxis,np.newaxis,:]
            else:
                continue

        return dataset

class HotPixelRemovalModule(Module):
    def initialise(self,windows,limit):
        self.windows = windows
        self.limit = limit
    def process(self,dataset,debug=False):
        for exp in range(dataset.num_exposures):
            wl, flux = dataset.wavelengths[exp,0,:], dataset.spec[exp,0,:]
            for window in self.windows:
                
                a = np.abs(wl-window[0])
                idx_low = np.where((a == np.min(a)))[0][0]
                a = np.abs(wl-window[1])
                idx_high = np.where((a == np.min(a)))[0][0]

                flux_section = flux[idx_low:idx_high]
                wave_section = wl[idx_low:idx_high]

                cont_model = np.poly1d(np.polyfit(wave_section,flux_section,3))
                subbed = flux_section - cont_model(wave_section)
                mask = ( subbed - np.nanmedian(subbed) > self.limit)
                flux[idx_low:idx_high] = scipy.interpolate.interp1d(wave_section[~mask],flux_section[~mask],'linear',fill_value='extrapolate')(wave_section)

            dataset.spec[exp,:,:] = flux[np.newaxis,np.newaxis,:]

        return dataset

class SubtractMasterTemplateModule(Module):
    def initialise(self,target,phase,weights=None):
        self.target = target
        self.phase = phase
        self.weights = weights
    def process(self,dataset,debug=False):
        phases = self.target.orbital_phase(dataset.obstimes)
        idx_oot = np.where((phases>self.phase) | (phases<-self.phase))[0]
        try:
            master_oot = np.average(dataset.spec[idx_oot,0,:],weights=self.weights[idx_oot],axis=0)
        except TypeError:
            master_oot = np.nanmean(dataset.spec[idx_oot,0,:],axis=0)
        dataset.spec[:,0,:] = dataset.spec[:,0,:]/master_oot[None,:]

        return dataset




