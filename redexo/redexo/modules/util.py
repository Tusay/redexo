__all__ = ['ShiftRestFrameModule','InjectSignalModule','InjectEmissionSpectrumModule','CoAddExposures', 'CoAddOrders', 'make_kp_vsys_map', 'highpass_gaussian','broaden','ConvolveToR']
from .base import Module
from ..core.dataset import Dataset, CCF_Dataset
import numpy as np
import astropy.constants as const
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import copy


def make_kp_vsys_map(ccf_map, Kp_list, target,in_transit=True,weights=None):
    snr_map = np.zeros((len(Kp_list), ccf_map.rv_grid.shape[-1]))
    mock_target = copy.deepcopy(target)
    for i,kp in enumerate(Kp_list):
        mock_target.Kp = kp
        restframe_ccf_map = ShiftRestFrameModule(target=mock_target)(ccf_map.copy())
        if in_transit:
            flat_ccf = CoAddExposures(weights=mock_target.in_transit(ccf_map.obstimes))(restframe_ccf_map)
        else:
            if weights is None:
                flat_ccf = CoAddExposures(weights=weights)(restframe_ccf_map)
            else:
                flat_ccf = CoAddExposures(weights=mock_target.orbital_phase(ccf_map.obstimes))(restframe_ccf_map)
        flat_ccf.normalize()
        snr_map[i] = flat_ccf.spec[0,0]
    return snr_map

class ShiftRestFrameModule(Module):
    def initialise(self, target=None, radial_velocities=None):
        if target is None and radial_velocities is None:
            raise ValueError("Provide either the target we are observing or an array with radial velocities")
        self.target = target
        self.rvs = radial_velocities

    def process(self, dataset, debug=False):
        if not self.target is None:
            self.rvs = self.target.radial_velocity(obs_time=dataset.obstimes)

        if isinstance(dataset, CCF_Dataset):
            for exp in range(dataset.num_exposures):
                for order in range(dataset.num_orders):
                    x = dataset.rv_grid[exp,order] - self.rvs[exp]
                    y = dataset.spec[exp,order]
                    dataset.spec[exp,order] = interp1d(x, y, assume_sorted=True, bounds_error=False)(dataset.rv_grid[exp, order])
        else:
            for exp in range(dataset.num_exposures):
                beta = 1-self.rvs[exp]/const.c.to('km/s').value
                new_wl = beta*dataset.wavelengths[exp]
                dataset.spec[exp] = interp1d(new_wl, dataset.spec[exp])(dataset.wavelengths)
        return dataset

class InjectSignalModule(Module):
    def initialise(self, template_wl, template, target=None, radial_velocities=None):
        self.template= template
        self.template_wl = template_wl
        self.target = target
        self.rvs = radial_velocities

    def process(self, dataset, debug=False):
        if self.rvs is None:
            self.rvs = self.target.radial_velocity(obs_time=dataset.obstimes)

        for exp in range(dataset.num_exposures):
            if self.target.in_transit(obs_time=dataset.obstimes[exp]):
                beta = 1-self.rvs[exp]/const.c.to('km/s').value
                wl_new = beta* dataset.wavelengths[exp]
                transit_depth = interp1d(self.template_wl, self.template, bounds_error=True, fill_value=np.median(self.template))(wl_new)
                dataset.spec[exp] *= transit_depth
        return dataset

class InjectEmissionSpectrumModule(Module):
    def initialise(self,template_wl,template,target=None,radial_velocities=None,phases=None,add_noise=False):
        self.template= template
        self.template_wl = template_wl
        self.target = target
        self.rvs = radial_velocities
        self.phases = phases
        self.add_noise = add_noise

    def process(self, dataset, debug=False):
        if self.rvs is None:
            self.rvs = self.target.radial_velocity(obs_time=dataset.obstimes)

        if self.phases is None:
            self.phases = np.ones(len(dataset.obstimes),dtype=bool)

        for exp in range(dataset.num_exposures):
            if self.phases[exp]:
                beta = 1-self.rvs[exp]/const.c.to('km/s').value
                wl_new = beta* dataset.wavelengths[exp]
                interpolated_template = interp1d(self.template_wl, self.template, bounds_error=True, fill_value=np.median(self.template))(wl_new)
                #multiply by the spectrum to get planet flux in ADU and add to flux

                if self.add_noise:
                    dataset.spec[exp] += interpolated_template * dataset.spec[exp] + np.random.normal(0,dataset.errors[exp])
                
                else:
                    dataset.spec[exp] += interpolated_template * dataset.spec[exp]
                
        return dataset
                


class CoAddExposures(Module):
    def initialise(self, indices=None, weights=None):
        self.weights= weights
        self.indices= indices

    def process(self, dataset, debug=False):
        if not dataset.errors is None:
            combined_errors = np.sqrt(np.sum(dataset.errors**2, axis=0))
        if self.weights is None:
            self.weights = np.ones(dataset.num_exposures)
        if self.indices is None:
            self.indices=np.arange(0,dataset.num_exposures)
        res = np.nansum((self.weights[self.indices]*dataset.spec[self.indices].T).T, axis=0)
        return dataset.__class__(res[np.newaxis,:], np.mean(dataset.wavelengths,axis=0)[np.newaxis,:], combined_errors[np.newaxis,:])

class CoAddOrders(Module):
    def initialise(self, weights=None):
        self.per_order_possible = False
        self.weights = weights
        if isinstance(self.weights, list):
            self.weights = np.array(self.weights)

    def process(self, dataset, debug=False):
        if not self.weights is None:
            weights = np.tile(self.weights[np.newaxis,:,np.newaxis], (dataset.num_exposures, 1, dataset.spec.shape[2]))
            co_added_spec = np.nansum(weights*dataset.spec, axis=1)[:,np.newaxis]
        else:
            co_added_spec = np.nansum(dataset.spec, axis=1)[:,np.newaxis]
        return dataset.__class__(co_added_spec, np.mean(dataset.wavelengths,axis=1)[:,np.newaxis], vbar=dataset.vbar, obstimes=dataset.obstimes, *dataset.header_info)


def highpass_gaussian(spec, sigma=100):
    continuum = gaussian_filter1d(spec, sigma)
    return spec/continuum


def broaden(wl, flux, sigma):
    return gaussian_filter1d(flux, sigma)


def ConvolveToR(wave, flux, R):
    mid_index = int(len(wave)/2)
    deltaWave = np.mean(wave)/ R
    newWavelengths = np.arange(np.log10(wave[0]), np.log10(wave[-1]), deltaWave)
    fwhm = deltaWave / (wave[mid_index] - wave[mid_index-1])
    std = fwhm / ( 2.0 * np.sqrt( 2.0 * np.log(2.0) ) ) #convert FWHM to a standard deviation
    g = Gaussian1DKernel(stddev=std)
    #2. convolve the flux with that gaussian
    flux = np.asarray(flux)
    convData = convolve(flux, g,boundary='extend')
    return wave, convData
