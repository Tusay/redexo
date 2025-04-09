__all__ = ["load_carmenes_data","load_harps_data","load_kpf_data","load_maroonx_data","load_espresso_data","load_parvi_data"]
import glob
from ..core import Dataset
import astropy
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord 
from astropy import units as u
import h5py
import numpy as np
import scipy
import sys

def sort_by_bjd(files,bjd_header_key,parvi=False):
    #takes list of fits files as input
    #returns that list sorted by BJD

    bjd = []
    for i in range(len(files)):
        data = fits.open(files[i])
        if parvi:
            bjd.append([data[1].header[bjd_header_key],i])
        else:
            bjd.append([data[0].header[bjd_header_key],i])
    bjd = np.array(sorted(bjd))

    return( [files[i] for i in bjd[:,1].astype(int)] )





def load_carmenes_data(folder, target=None, spectral_orders=None, skip_exposures = [], header_info={}):
    files = glob.glob(folder+'*nir_A.fits')
    print('Loading {0} files...'.format(len(files)-len(skip_exposures)))
    files = sorted(files)
    dataset = Dataset(target)

    for i,fname in enumerate(files):
        if not i in skip_exposures:
            data = fits.open(fname)

            vbar = data[0].header['HIERARCH CARACAL BERV']
            if spectral_orders is None:
                wl = data[4].data
                spectrum = data[1].data
                errors = data['SIG'].data
            else:
                wl = data[4].data[spectral_orders]
                spectrum = data[1].data[spectral_orders]
                errors = data['SIG'].data[spectral_orders]

            BJD = (data[0].header['HIERARCH CARACAL BJD'] + 2400000)

            header_data = {str(key): data[0].header[val] for key, val in header_info.items()}
            dataset.add_exposure(spectrum, wl=wl, errors=errors, vbar=vbar, obstime=BJD, exp_num=i, **header_data)
    return dataset

def load_maroonx_data(file_path,target=None,spectral_orders=None):

    #This utilizes the HDF format of the reduced spectra
    f = h5py.File(file_path,'r')
    num_files = f['template']['tstack_f'].shape[1]
    dataset = Dataset(target)

    for i in range(num_files):
        
        vbar = f['rv']['berv'][i]
        if spectral_orders is None:
            wl = f['template']['tstack_w'][:,i,:]
            spectrum = f['template']['tstack_f'][:,i,:]
            errors = 0.001*np.ones(spectrum.shape) #idk where the flux error info is
        else:
            wl = f['template']['tstack_w'][spectral_orders,i,:]
            spectrum = f['template']['tstack_f'][spectral_orders,i,:]
            errors = 0.001*np.ones(spectrum.shape)

        BJD = f['rv']['bjd'][i]

        dataset.add_exposure(spectrum,wl=wl,errors=errors,vbar=vbar,obstime=BJD)

    return dataset


def load_harps_data(folder,target=None,skip_exposures=[],spectral_orders=None,header_info={},TAC=True,mask_tellurics=False,cut_off=0.4,which_spectrograph='south'):
    files = glob.glob(folder+'*_formatted_TAC.fits')
    print('Loading {0} files...'.format(len(files)-len(skip_exposures)))
    #files = sorted(files)
    
    if which_spectrograph == 'south':
        files = sort_by_bjd(files,'HIERARCH ESO DRS BJD')
    elif which_spectrograph == 'north':
        files = sort_by_bjd(files,'BJD')

    #dataset = Dataset(spec=None,wavelengths=None,errors=None,target=target,header_info=header_info)
    #dataset = Dataset(target=target,header_info=header_info)
    
    dataset = Dataset(target)
    #dataset = Dataset(spec=None,wavelengths=None,errors=None,vbar=[],obstimes=[],header_info=header_info)
    #get the length of the longest array in the data
    lengths = []
    for i in files:
        lengths.append( len(fits.open(i)[1].data.WAVE) )

    max_length = np.max(lengths)



    if mask_tellurics:
        mask = np.full((len(files), max_length), False)

        for i,fname in enumerate(files):
            if not i in skip_exposures:
                
                data = fits.open(fname)
                spectrum = data[1].data.tacflux
                
                a = np.abs(data[1].data.WAVE-5290.0)
                idx_low = np.where((a == np.min(a)))[0][0]
                a = np.abs(data[1].data.WAVE-5320.0)
                idx_high = np.where((a == np.min(a)))[0][0]
                if which_spectrograph=='south': #remove the detector chip gap in HARPS data. Set north for HARPSN
                    gap_idx = np.where( np.logical_and((np.logical_and(data[1].data.WAVE<5340.0,data[1].data.WAVE>5290.0)),(data[1].data.tacflux==0.0)))[0]
                    spectrum[gap_idx] = np.nan

                mtrans = np.pad(data[1].data.mtrans, (0, max_length - len(data[1].data.mtrans)), 'constant', constant_values=(0,0)) #pad the transmission mask with zeroes
                spectrum = np.pad(spectrum,(0, max_length - len(spectrum)), 'constant',constant_values=(0,0))
                #spectrum = np.pad(data[1].data.tacflux,(0, max_length - len(data[1].data.tacflux)), 'constant',constant_values=(0,0))
                
                mask[i,:] = mask[i,:] + (mtrans < cut_off)+(spectrum==0)
                
        
        mask = ~mask.any(axis=0)
        
    for i,fname in enumerate(files):
        if not i in skip_exposures:
            data = fits.open(fname)
            
            if which_spectrograph=='south':
                vbar = data[0].header['HIERARCH ESO DRS BERV']
            elif which_spectrograph=='north':
                vbar = data[0].header['BERV']

            if TAC==True:
                if spectral_orders is None:
                    wl = np.pad(data[1].data.WAVE,(0, max_length - len(data[1].data.WAVE)), 'edge')
                    wl = wl[mask][np.newaxis,:]

                    spectrum = data[1].data.tacflux

                    a = np.abs(data[1].data.WAVE-5290.0)
                    idx_low = np.where((a == np.min(a)))[0][0]
                    a = np.abs(data[1].data.WAVE-5320.0)
                    idx_high = np.where((a == np.min(a)))[0][0]

                    gap_idx = np.where( np.logical_and((np.logical_and(data[1].data.WAVE<5340.0,data[1].data.WAVE>5290.0)),(data[1].data.tacflux==0.0)))[0]
                    spectrum[gap_idx] = np.mean(data[1].data.tacflux[idx_low:idx_high])





                    #spectrum = np.pad(data[1].data.tacflux,(0, max_length - len(data[1].data.tacflux)), 'constant',constant_values=(0,0))
                    spectrum = np.pad(spectrum,(0, max_length - len(spectrum)), 'constant',constant_values=(0,0))
                    spectrum = spectrum[mask][np.newaxis,:]
                    
                    errors = np.ones(len(wl[0]))[np.newaxis,:]
                else:
                    wl = data[1].data.WAVE[spectral_orders] #nothing in this loop works past this point lmao
                    spectrum = data[1].data.FLUX[spectral_orders]
                    errors = np.ones(len(wl[0]))
            else:
                if spectral_orders is None:
                    wl = data[1].data.WAVE[np.newaxis,:]
                    spectrum = data[1].data.FLUX[np.newaxis,:]
                    errors = np.ones(len(wl[0]))[np.newaxis,:]
                else:
                    wl = data[1].data.WAVE[spectral_orders]
                    spectrum = data[1].data.FLUX[spectral_orders]
                    errors = np.ones(len(wl[0]))

            if which_spectrograph=='south':
                BJD = (data[0].header['HIERARCH ESO DRS BJD'])
            elif which_spectrograph=='north':
                BJD = (data[0].header['BJD'])
            #header_data = {str(key): data[0].header[val] for key, val in header_info.items()}
            header_data = {str(key): data[0].header[key] for key, val in header_info.items()}
            dataset.add_exposure(spectrum=spectrum, wl=wl, errors=errors, vbar=vbar, obstime=BJD, exp_num=i, **header_data)
    return dataset


def load_kpf_data(folder,star_name=None,target=None,skip_exposures=[],spectral_orders=None,header_info={},TAC=True,mask_tellurics=False,cut_off=0.4):

    if star_name is None:
        print("Provide a Simbad-resolvable star name")
        sys.exit()

    files = glob.glob(folder+'*formatted_TAC.fits')
    print('Loading {0} files...'.format(len(files)-len(skip_exposures)))
    files = sort_by_bjd(files,'MJD-OBS',parvi=False)
    dataset = Dataset(spec=None,wavelengths=None,errors=None,target=target)

    #make master telluric template
    if mask_tellurics:
        mask = np.full((len(files), len(fits.open(files[0])[1].data.WAVE)), False)

        for i,fname in enumerate(files):
            if not i in skip_exposures:
                data = fits.open(fname)
                mask[i,:] = mask[i,:] + (data[1].data.mtrans > cut_off)

        mask = mask.any(axis=0)



    for i,fname in enumerate(files):
        if not i in skip_exposures:
            data = fits.open(fname)

            #get the barcycentric velocity, temporary until this is populated in the header.
            sc = SkyCoord.from_name(star_name)
            loc = astropy.coordinates.EarthLocation.of_site('Keck')
            t_obs = Time(Time(data[0].header['DATE-BEG']))
            #t_obs = Time(data[0].header['MJD-OBS']+2400000.5,format='jd')
            vbar_calc = sc.radial_velocity_correction(obstime=t_obs, location=loc)
            vbar = vbar_calc.to(u.km/u.s).value

            if TAC==True:
                if spectral_orders is None:
                    wl = data[1].data.WAVE[mask][np.newaxis,:]
                    spectrum = data[1].data.tacflux[mask][np.newaxis,:]
                    errors = np.ones(len(wl[0]))[np.newaxis,:]#again no flux errors
                else:
                    wl = data[1].data.WAVE[spectral_orders]
                    spectrum = data[1].data.FLUX[spectral_orders]
                    errors = np.ones(len(wl[0]))
            else:
                if spectral_orders is None:
                    wl = data[1].data.WAVE[np.newaxis,:]
                    spectrum = data[1].data.FLUX[np.newaxis,:]
                    errors = np.ones(len(wl[0]))[np.newaxis,:]#again no flux errors
                else:
                    wl = data[1].data.WAVE[spectral_orders]
                    spectrum = data[1].data.FLUX[spectral_orders]
                    errors = np.ones(len(wl[0]))

            header_data = {str(key): data[0].header[val] for key, val in header_info.items()}
            dataset.add_exposure(spectrum=spectrum, wl=wl, errors=errors, vbar=vbar, obstime=t_obs.jd,exp_num=i, **header_data)
    return dataset


def load_espresso_data(folder,target=None,planet=None,skip_exposures=[],spectral_orders=None,header_info={},TAC=True,mask_tellurics=False,cut_off=0.4):
    files = glob.glob(folder+'*_formatted_TAC.fits')
    print('Loading {0} files...'.format(len(files)-len(skip_exposures)))
    files = sorted(files)
    dataset = Dataset(spec=None,wavelengths=None,errors=None,target=target,planet=planet)
    dataset.clear_data()
    
    #make master telluric template
    if mask_tellurics:
        mask = np.full((len(files), len(fits.open(files[0])[1].data.WAVE)), False)

        for i,fname in enumerate(files):
            if not i in skip_exposures:
                data = fits.open(fname)
                mask[i,:] = mask[i,:] + (data[1].data.mtrans > cut_off)

        mask = mask.any(axis=0)

    for i,fname in enumerate(files):
        if not i in skip_exposures:
            data = fits.open(fname)
            
            vbar = data[0].header['HIERARCH ESO QC BERV']
            if TAC==True:
                if spectral_orders is None:
                    wl = data[1].data.WAVE[mask][np.newaxis,:]
                    spectrum = data[1].data.tacflux[mask][np.newaxis,:]
                    errors = np.ones(len(wl[0]))[np.newaxis,:]#again no flux errors
                else:
                    wl = data[1].data.WAVE[spectral_orders]
                    spectrum = data[1].data.FLUX[spectral_orders]
                    errors = np.ones(len(wl[0]))
            else:
                if spectral_orders is None:
                    wl = data[1].data.WAVE[np.newaxis,:]
                    spectrum = data[1].data.FLUX[np.newaxis,:]
                    errors = np.ones(len(wl[0]))[np.newaxis,:]#again no flux errors
                else:
                    wl = data[1].data.WAVE[spectral_orders]
                    spectrum = data[1].data.FLUX[spectral_orders]
                    errors = np.ones(len(wl[0]))

            BJD = (data[0].header['HIERARCH ESO QC BJD'])
            
            header_data = {str(key): data[0].header[val] for key, val in header_info.items()}
            dataset.add_exposure(spectrum=spectrum, wl=wl, errors=errors, vbar=vbar, obstime=BJD,exp_num=i, **header_data)
    return dataset


def load_parvi_data(folder, target=None, spectral_orders=None, skip_exposures = [], normalize=True,header_info={}):
    print("Did you remember to change the object name?")
    files = glob.glob(folder+'*.fits')
    print('Loading {0} files...'.format(len(files)-len(skip_exposures)))
    #files = sorted(files)
    files = sort_by_bjd(files,'BARYJD',parvi=True)
    dataset = Dataset(target)

    sc = SkyCoord.from_name('HD 189733')
    #sc = SkyCoord.from_name('HD 201033')
    #sc = SkyCoord.from_name('HD 195689') #kelt 9
    loc = astropy.coordinates.EarthLocation.of_site('palomar')
    if normalize:
        flux_idx = 3
        flux_err_idx = 4
    else:
        flux_idx = 1
        flux_err_idx = 2


    for i,fname in enumerate(files):
        if not i in skip_exposures:
            data = fits.open(fname)
            
            #all of this should be in data headers, not calculated here this is dumb.
            jd_start = Time(float(data[1].header['TIMEI00'])/1E9, format='unix').jd
            exp_time = data[1].header['EXPTIME']/86400
            jd_mid = jd_start + exp_time/2
            #t_obs = Time(jd_mid, format='jd')
            t_obs = Time(data[1].header['BARYJD'], format='jd')
            vbar_calc = sc.radial_velocity_correction(obstime=t_obs, location=loc)
            vbar = vbar_calc.to(u.km/u.s).value
            print(vbar)
            if spectral_orders is None:
                
                print("Not working right now, please specify which orders you want to load")
                exit()

            else:
                nan_idx = np.argwhere(~np.isnan(data[1].data[f"{spectral_orders[0]}.3"][flux_idx]))
                wl = np.zeros((len(spectral_orders),len(nan_idx)))
                spectrum = np.zeros((len(spectral_orders),len(nan_idx)))
                errors = np.zeros((len(spectral_orders),len(nan_idx)))
                #spectrum = np.zeros((len(spectral_orders),data[1].data[0][0].shape[0]))
                #errors = np.zeros((len(spectral_orders),data[1].data[0][0].shape[0]))

                for j,order in enumerate(spectral_orders):
                    
                    wl[j] = 10*data[1].data[f"{order}.3"][0][nan_idx].reshape(len(nan_idx)) #.3 suffix indicates the science fiber
                    spectrum[j,:] = data[1].data[f"{order}.3"][flux_idx][nan_idx].reshape(len(nan_idx))
                    errors[j,:] = data[1].data[f"{order}.3"][flux_err_idx][nan_idx].reshape(len(nan_idx))

            
            header_data = {str(key): data[0].header[val] for key, val in header_info.items()}
            dataset.add_exposure(spectrum, wl=wl, errors=errors, vbar=vbar, obstime=jd_mid, exp_num=i, **header_data)
    return dataset







def load_crires_data(folder):
    raise NotImplementedError


