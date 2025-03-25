__all__ = ['Dataset', 'CCF_Dataset']
import numpy as np
import copy
import matplotlib.pyplot as plt

class Dataset(object):
    def __init__(self, spec=None, wavelengths=None, errors=None, vbar=[], obstimes=[], exp_num=[], **kwargs):
        '''
        Main dataset class that keeps the spectra
        spec: 

        '''
        self.header_info = dict()
        
        
        for key, value in kwargs.items():
            
            self.header_info[key] = value
        self._vbar = vbar
        self._obstimes = obstimes
        self._exp_num = exp_num
        
        #Also allow for directly assigning spec, wavelengths and errors
        #Make sure that the array always has dimensions (exposures, orders, wavelength)
        if spec is not None:
            if spec.ndim ==2:
                self.spec = spec[:, np.newaxis, :] 
            else:
                self.spec = spec

            if wavelengths.ndim==1:
                self.wavelengths = np.tile(wavelengths[np.newaxis, np.newaxis, :], (self.num_exposures, self.num_orders, 1))
            elif wavelengths.ndim == 2:
                self.wavelengths = np.tile(wavelengths[np.newaxis,:, :], (self.num_exposures, 1, 1))
            else:
                self.wavelengths = wavelengths
            if errors is not None:
                self.errors = errors
            else:
                self.errors = np.ones_like(self.spec)
    
    @property
    def num_orders(self):
        return self.spec.shape[1]

    @property
    def num_exposures(self):
        return self.spec.shape[0]

    @property
    def phases(self):
        return self.target.orbital_phase(self.obstimes)

    @property
    def transit_indices(self):
        return np.where(self.target.in_transit(self.phases))[0]

    @property
    def planet_rvs(self):
        return (-self.vbar*u.km/u.s)+self.target.planet_rv(self.phases)

    @property
    def obstimes(self):
        return np.array(self._obstimes)

    @property
    def vbar(self):
        return np.array(self._vbar)

    @property
    def exp_num(self):
        return np.array(self._exp_num)

    def clear_data(self):
        # List all attributes and check their types
        attributes = vars(self)
        for attr_name, attr_value in attributes.items():
            if isinstance(attr_value, list):
                # print(f"Clearing list: {attr_name}")
                setattr(self, attr_name, [])  # Reset the list

    def add_exposure(self, spectrum, wl=None, errors=None, obstime=None, vbar=None, exp_num=None, **kwargs):
        '''

        kwargs: Use this when you want to store more data regarding the exposures, e.g. airmass or signal-to-noise ratio
                This will be stored in the header_info dictionary
        '''
        if not hasattr(self, 'spec'):
            assert spectrum.ndim==2, "Flux should have shape (num_orders, wavelength_bins_per_order)"
            self.spec = spectrum[np.newaxis,:]
            self.wavelengths = wl[np.newaxis,:]
            if not errors is None:
                self.errors = errors[np.newaxis,:]
            else:
                self.errors = np.ones_like(self.spec)
        else:
            assert spectrum.shape==self.spec.shape[1:], "Added flux should have shape (num_orders, wavelength_bins_per_order)"
            self.spec = np.concatenate([self.spec, spectrum[np.newaxis,:]],axis=0)
            self.wavelengths = np.concatenate([self.wavelengths, wl[np.newaxis,:]],axis=0)
            if not errors is None:
                self.errors = np.concatenate([self.errors, errors[np.newaxis,:]],axis=0)

        if not obstime is None:
            self._obstimes.append(obstime)
        if not vbar is None:
            self._vbar.append(vbar)
        if not exp_num is None:
            self._exp_num.append(exp_num)
        
        for key, value in kwargs.items():
            self.header_info[key].append(value)
            #self.header_info[key] = value        
    def drop_exposures(self, drop_indices):
        print(drop_indices)
        if not isinstance(drop_indices, list) or not isinstance(drop_indices, np.ndarray):
            indices = [drop_indices]
        to_not_drop = [idx for idx in np.arange(self.num_exposures) if idx not in drop_indices ]
        self.spec = self.spec[to_not_drop]
        self.wavelengths = self.wavelengths[to_not_drop]
        self.errors = self.errors[to_not_drop]

        self._vbar = [self._vbar[idx] for idx in to_not_drop]
        self._obstimes = [self._obstimes[idx] for idx in to_not_drop]
        self._exp_num = [self._exp_num[idx] for idx in to_not_drop]

    def set_results(self, spec, wavelengths=None, errors=None, order=None):
        if isinstance(spec, Dataset) or isinstance(spec, CCF_Dataset):
            wavelengths = spec.wavelengths
            errors = spec.errors
            spec = spec.spec
        else:
            if order is None:
                if not spec.shape==self.spec.shape:
                    raise ValueError("The spec you are trying to set has an incorrect shape, expected: {0}, received: {1}", self.spec.shape, spec.shape)
                order = np.arange(self.num_orders)
        
        self.spec[:, order:order+1, :] = spec
        if wavelengths is not None: 
            self.wavelengths[:, order:order+1, :] = wavelengths
        if errors is not None:
            self.errors[:, order:order+1, :] = errors

    def get_order(self, order, as_dataset=True):
        if not as_dataset:
            return self.spec[:, order], self.wavelengths[:, order], self.errors[:, order]
        else:
            return self.__class__(self.spec[:,order,np.newaxis], self.wavelengths[:,order,np.newaxis], self.errors[:, order, np.newaxis], self.vbar, self.obstimes, *self.header_info)
        
    def make_clean(self, shape):
        self.spec = np.zeros(shape)
        self.wavelengths = np.zeros(shape)
        self.errors = np.zeros(shape)

    def copy(self):
        return copy.deepcopy(self)

    def plot_river(self):
        fig,ax = plt.subplots(figsize=(21,int(self.num_exposures/4)))
        # print(self.wavelengths[0][0][0],self.wavelengths[0][0][-1],self.num_exposures)
        im=ax.imshow(self.spec[:,0,:],aspect='auto',cmap='bwr',origin='lower',extent=[self.wavelengths[0][0][0],self.wavelengths[0][0][-1],self.num_exposures,0])
        fig.colorbar(im)
        ax.set_xlabel(r'Wavelength [$\AA$]')
        ax.set_ylabel('Exposure Number')
        ax.tick_params(top=True, bottom=True, left=True, right=True,direction='in')
        return fig

class CCF_Dataset(Dataset):
    def __init__(self, spec=None, rv_grid=None, vbar=[], obstimes=[], **kwargs):
        super().__init__(spec=spec, wavelengths=rv_grid, vbar=vbar, obstimes=obstimes, **kwargs)

    @property
    def rv_grid(self):
        return self.wavelengths

    @rv_grid.setter
    def rv_grid(self, rv_grid):
        self.wavelengths = rv_grid

    def get_order(self, order, as_dataset=True):
        if not as_dataset:
            return self.spec[:, order], self.rv_grid[:, order], self.errors[:, order]
        else:
            return CCF_Dataset(self.spec[:,order,np.newaxis], self.rv_grid[:,order,np.newaxis], self.vbar, self.obstimes, *self.header_info)
        
    def normalize(self, exclude_region=50):
        mask = np.abs(self.rv_grid[0,0])>exclude_region
        means = np.tile(np.nanmean(self.spec[:,:,mask],axis=-1)[:,:,np.newaxis], (1,1,self.spec.shape[-1]))
        stds = np.tile(np.nanstd(self.spec[:,:,mask],axis=-1)[:,:,np.newaxis], (1,1,self.spec.shape[-1]))
        self.spec = (self.spec-means)/stds
        return self

'''
    def subtract_model(self,model,optimize=False,exclude_region=None):

        """
        Function for subtracting a model CCF (e.g empirical RM models). 

        Parameters:
            model (array): CCF matrix with same dimensions as working CCF.
            optimize (bool): Whether to optimize the model with a simple scaling value 
            by minimizing the sum of squared residuals.
            exclude_region (array): Region of CCF space to exclude when calculating residuals. 

        Returns:
            CCF Dataset
        """
        
        if self.spec.shape != model.shape:
            print(f"Model and CCF Dataset have different dimensions: {self.spec.shape} =/= {model.shape}")
            exit()

        if not optimize:
            self.spec = self.spec - model
            return self

        else:
            if not exclude_region:
'''

