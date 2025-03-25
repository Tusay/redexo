__all__ = ['SysRemModule', 'PCASubtractModule']
from .base import Module
import numpy as np
import matplotlib.pyplot as plt


class PCASubtractModule(Module):

    def initialise(self, number_of_modes, rcond=1e-5):
        self.number_of_modes = number_of_modes
        self.per_order = True

    def process(self, dataset, debug=False):
        spec = dataset.spec[:,0,:]
        norm_spec = (spec- np.nanmean(spec, axis=0))
        nans = np.any(np.isnan(norm_spec),axis=0)
        norm_spec[:,nans] = np.nan
        u,s,v = np.linalg.svd(norm_spec[:,~nans], full_matrices=False)
        s_new = np.copy(s)
        s_new[:self.number_of_modes] = 0

        spec_corr = np.dot(u, np.dot(np.diag(s_new), v))
        norm_spec[:, ~nans] = spec_corr
        dataset.spec = norm_spec[:,np.newaxis]
        return dataset

class SysRemModule(Module):
    '''
    mode: subtract/divide
    '''
    def initialise(self, number_of_modes, max_iterations_per_mode=1000, mode='subtract'):
        self.number_of_modes = number_of_modes
        self.max_iterations_per_mode = max_iterations_per_mode
        self.per_order = True
        self.mode = mode

    def process(self, dataset, debug=False):
        spec = dataset.spec[:,0,:]
        errors = dataset.errors[:,0,:]
        spec = (spec.T - np.nanmean(spec, axis=1)).T
        a = np.ones(dataset.num_exposures)
        nans = np.any(np.isnan(spec), axis=0)
        spec = spec[:,~nans]
        errors = errors[:, ~nans]

        for i in range(self.number_of_modes):
            correction = np.zeros_like(spec)
            for j in range(self.max_iterations_per_mode):
                prev_correction = correction
                c = np.nansum(spec.T *a/errors.T**2, axis=1) / np.nansum(a**2/errors.T**2, axis=1)
                a = np.nansum(c*spec/errors**2, axis=1) / np.nansum(c**2/errors**2, axis=1)
                correction = np.dot(a[:,np.newaxis], c[np.newaxis,:])
                #spec -= correction
                fractional_dcorr = np.sum(np.abs(correction-prev_correction))/(np.sum(np.abs(prev_correction))+1e-5)
                if j>1 and fractional_dcorr< 1e-3:
                    break
            if self.mode=='subtract':
                spec -= correction
            elif self.mode=='divide':
                spec /= correction
                errors /= correction
        full_spec = np.tile(np.nan, dataset.spec[:,0,:].shape)
        full_spec [:,~nans] = spec
        dataset.spec = full_spec[:,np.newaxis]
        return dataset