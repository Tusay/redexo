__all__ = ['CrossCorrelationModule']
from .base import Module
from ..core.dataset import CCF_Dataset
import astropy.constants as const
from scipy.interpolate import interp1d
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import copy

def human_time(deltat):
    if deltat>86400:
        return deltat/86400,"days"
    elif deltat>3600:
        return deltat/3600,"hours"
    elif deltat>60:
        return deltat/60,"minutes"
    else:
        return deltat,"seconds"


class CrossCorrelationModule(Module):
    def initialise(self, template, template_wl, rv_range, drv, error_weighted=False, mask_rm=False,rm_mask_range=None,rm_mask_value=0.0,subtract_oot=False,target=None,phase=None):
    # def __init__(self, template, template_wl, rv_range, drv, error_weighted=False, mask_rm=False,rm_mask_range=None,rm_mask_value=0.0,subtract_oot=False,target=None,phase=None):
        self.target = target
        self.phase = phase
        self.rv_grid = None  # Clear rv_grid if it persists
        self.beta_grid = None  # Clear beta_grid if it persists
        # Initialize other attributes from scratch
        self.template = template
        self.template_wl= template_wl
        self.rv_range = rv_range
        self.drv = drv
        self.rv_grid = np.arange(-rv_range, rv_range+1e-4, drv)
        self.beta_grid = 1.0-self.rv_grid/const.c.to('km/s').value
        self.error_weighted = error_weighted
        self.mask_rm = mask_rm
        self.rm_mask_range = rm_mask_range
        self.rm_mask_value = rm_mask_value
        self.subtract_oot = subtract_oot
        # print(f"CrossCorrelationModule ID: {id(self)}")
        # print(f"Target object ID: {id(self.target)}")

    def process(self, dataset, plot_stellar_rv=False, debug=False):
        CCF = np.zeros((dataset.num_exposures, dataset.num_orders, self.rv_grid.size))
        phases = np.zeros_like(dataset.obstimes)  # Reset phases array
        # start_time=time.time()
        # print(f"Starting CCF loop on {dataset.num_exposures} exposures...")
        for exp in range(dataset.num_exposures):
            nans = np.isnan(dataset.spec[exp])
            wl_exp = dataset.wavelengths[exp][~nans]
            spec_exp = dataset.spec[exp][~nans]
            errors_exp = dataset.errors[exp][~nans]
            shifted_wavelengths = wl_exp*self.beta_grid[:,np.newaxis]
            
            shifted_template = interp1d(self.template_wl, self.template, bounds_error=True)(shifted_wavelengths)
            shifted_template = shifted_template.T - np.mean(shifted_template, axis=1)
            if self.error_weighted:
                CCF[exp] = (spec_exp/errors_exp**2).dot(shifted_template)
            else:
                CCF[exp] = spec_exp.dot(shifted_template)
        # end_time,end_label=human_time(time.time()-start_time)
        # print(f"The CCF loop took {end_time:.2f} {end_label}.")

        def get_fwhm(spec):
            avg_spec = spec.mean(axis=0)
            hm=(avg_spec.min() + avg_spec.max())/2
            idx1=(np.abs(avg_spec - hm)).argmin()
            idx2=(np.abs(np.array(list(avg_spec[:idx1-1])+list(avg_spec[idx1+2:])) - hm)).argmin()
            fwhm = abs(full_rv_matrix[0][0][idx1]-full_rv_matrix[0][0][idx2])
            return fwhm, idx1, idx2

        if plot_stellar_rv==True:
            full_star_data=dataset.copy()
            full_CCF = copy.deepcopy(CCF)
            rv_matrix = np.tile(self.rv_grid[np.newaxis, np.newaxis,:], (dataset.num_exposures, dataset.num_orders, 1))
            full_rv_matrix = copy.deepcopy(rv_matrix)
            full_res = CCF_Dataset(spec=full_CCF, rv_grid=full_rv_matrix, vbar=full_star_data.vbar, obstimes=full_star_data.obstimes)
            fwhm,idx1,idx2=get_fwhm(full_res.spec[:,0,:]*1e6)
            fig,ax = plt.subplots(figsize=(21,int(full_star_data.num_exposures/4)))
            im=ax.imshow(full_res.spec[:,0,:]*1e6,aspect='auto',extent=[full_rv_matrix[0][0][0],full_rv_matrix[0][0][-1],0,full_star_data.num_exposures])
            fig.colorbar(im)
            ax.axvline(full_rv_matrix[0][0][idx1],color='r',linestyle='dashed')
            ax.axvline(full_rv_matrix[0][0][idx2],color='r',linestyle='dashed')
            print(f"FWHM: {fwhm:.1f} km/s")
            ax.set_xlabel('RV [km/s]')
            ax.set_ylabel('Exp. #')
            plt.show()

        # print(f"self.target: {self.target}")
        if self.subtract_oot:
            # start_time=time.time()
            # print(f"Dividing out the out-of-transit exposures...")        
            try:
                # print("Starting CrossCorrelationModule process...")
                # print(f"dataset: {dataset}")
                # print(f"obstimes: {dataset.obstimes}")  # Print obstimes to verify its value
                if dataset.obstimes is None or len(dataset.obstimes) == 0:
                    raise ValueError("Error: dataset.obstimes is None or empty.")
                # print(f"self.target: {self.target}")
                if self.target is None:
                    raise ValueError("Error: self.target is None. Did you forget to set it?")
                if not hasattr(self.target, "orbital_phase"):
                    raise AttributeError("Error: self.target does not have an orbital_phase method.")
                # print(self.target.orbital_phase(dataset.obstimes[0]))
                phases = np.array([self.target.orbital_phase(time) for time in dataset.obstimes])
                # print(f"Computed phases (manual loop): {phases}")
                phases = self.target.orbital_phase(dataset.obstimes)
                # print(f"Phases: {phases}")  # Print computed phases
                if phases is None or len(phases) == 0:
                    raise ValueError("Computed phases are None or empty.")
                # print(f"self.phase: {self.phase}")
                phase_idx = np.where( (phases>self.phase) | (phases<-self.phase))[0]
                # print(f"phase_idx: {phase_idx}")
                if len(phase_idx) == 0:
                    raise ValueError("No indices found for phase cut-off. Check self.phase.")
                # print(f"CCF shape: {CCF.shape}")
                avg_oot=np.nanmean(CCF[phase_idx,0,:],axis=0)
                # print(f"avg_oot: {avg_oot}")  # Check if it's NaN or empty
                denominator = avg_oot[None, :] + np.max(CCF[:, 0, :])
                # print(f"Denominator min/max: {np.min(denominator)}, {np.max(denominator)}")
                CCF = ((CCF[:,0,:]+np.max(CCF[:,0,:]))/(avg_oot[None,:]+np.max(CCF[:,0,:])))-1.0 #adding max values to prevent Div0 errors.
                CCF = CCF[:,np.newaxis,:]
            except Exception as e:
                print(f"Exception occurred: {e}")
                sys.exit(1)
            # except:
            #     print("None value. Please provide both a target and phase cut-off")
            #     sys.exit(1)
            # end_time,end_label=human_time(time.time()-start_time)
            # print(f"Dividing out the OOT exposures took {end_time:.2f} {end_label}.")

        if self.mask_rm:
            try:
                rv_idx = np.where( (self.rv_grid>self.rm_mask_range[0]) & (self.rv_grid<self.rm_mask_range[1]) )[0]
                CCF[:,0,rv_idx] = self.rm_mask_value
            except Exception as e:
                print(f"Exception occurred: {e}")
                sys.exit(1)
            # except:
            #     print("None value. Please provide both a mask range and mask value")
            #     sys.exit(1)

        
        
        rv_matrix= np.tile(self.rv_grid[np.newaxis, np.newaxis,:], (dataset.num_exposures, dataset.num_orders, 1))
        # print(f"rv_matrix: {rv_matrix}")
        res = CCF_Dataset(spec=CCF, rv_grid=rv_matrix, vbar=dataset.vbar, obstimes=dataset.obstimes)
        
                
        return res
