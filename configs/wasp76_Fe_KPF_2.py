# %%
# Imports
import gc
import copy
import time
import numpy as np
import pandas as pd
import scipy 
import matplotlib.pyplot as plt
import os, glob, sys
from tqdm import tqdm
from astropy import units as u
from astropy.io import fits
from astropy import convolution
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import pickle
from PyAstronomy import pyasl

sys.path.append('../redexo')
from redexo import *

params={"axes.titlesize": 26,"axes.labelsize": 26,"axes.linewidth": 3,"axes.axisbelow": True,
"lines.linewidth": 3,"lines.markersize": 10,
"xtick.labelsize": 24,"xtick.top": True,"xtick.major.size": 10,"xtick.minor.size": 5,
"xtick.major.width": 5,"xtick.minor.width": 3,"xtick.major.pad": 15,"xtick.direction": 'in',"xtick.minor.visible": True,
"ytick.labelsize" : 24,"ytick.right": True,"ytick.major.size": 10,"ytick.minor.size": 5,
"ytick.major.width": 5,"ytick.minor.width": 3,"ytick.major.pad": 15,"ytick.direction": 'in',
"ytick.minor.visible": True,
"legend.fontsize": 24,"legend.borderaxespad": 1.0}
plt.rcParams.update(params)

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

def human_time(deltat):
    if deltat>86400:
        return deltat/86400,"days"
    elif deltat>3600:
        return deltat/3600,"hours"
    elif deltat>60:
        return deltat/60,"minutes"
    else:
        return deltat,"seconds"

def hp_filter(spec,num_exposures,size=100,plot1D=False):
    fixed_spec=[]
    if plot1D==True:
        x = np.linspace(-(len(spec[0])-1)/2, (len(spec[0])-1)/2, len(spec[0]))
        fig,ax=plt.subplots(figsize=(21,6))
        spec1D = np.zeros(len(spec[0]))
    for exp in range(num_exposures):
        # data2D=old_ccf_map_earth_spec[:,0,:]*1e6
        # data2D=ccf_map_earth.spec[:,0,:]*1e6
        offset=abs(np.nanmin(spec[exp]))
        smoothed = savgol_filter(spec[exp]+2*offset, size, 3, mode='interp') 
        fixed_spec.append((spec[exp]+2*offset)-smoothed)
        if plot1D==True:
            spec1D+=(spec[exp]+2*offset)
            plt.plot(x,spec[exp]+2*offset)
        # fixed_spec.append((data2D[exp]+2*offset)/smoothed-1)
    if plot1D==True:
        # plt.plot(x,spec1D/num_exposures)
        plt.plot(x,smoothed)
        plt.show()
    return np.asarray(fixed_spec)[:,np.newaxis,:]

def plot_exposures(data2D,dataset,planet,xextent=None,xlabel=None,separate_transit=False,zoom_in=None,line_by_line=False):
    phases = planet.orbital_phase(dataset.obstimes)
    transit_idxs = np.where(np.logical_and(phases>-planet.transit_start, phases<planet.transit_start))[0]
    oot_idxs = [i for i in range(len(phases)) if i not in transit_idxs]
    oot_pre = [transit_idxs[0]-1 if transit_idxs[0]-1 >= 0 else None][0]
    oot_post = [transit_idxs[-1]+1 if transit_idxs[-1]+1 <= len(phases)-1 else None][0]
    if xextent is None:
        xextent = data2D[0]
    if line_by_line==True:
        for exp in range(dataset.num_exposures):
            if len(xextent)<len(data2D[0]):
                xextent=np.linspace(xextent[0],xextent[-1],len(data2D))
            fig,ax=plt.subplots(figsize=(21,6))
            plt.plot(xextent,data2D[exp],label=f'Exp #{exp}')
            plt.xlabel=xlabel
            plt.legend()
            plt.show()
        return None
    if separate_transit==True:
        fig,ax = plt.subplots(nrows=2,figsize=(21,int(dataset.num_exposures/4)))
        extent0=[xextent[0],xextent[-1],transit_idxs[0],transit_idxs[-1]]
        im1=ax[1].imshow(data2D[transit_idxs],aspect='auto',extent=extent0)
        fig.colorbar(im1)
        extent1=[xextent[0],xextent[-1],[oot_pre if oot_pre!=None else 0][0],[oot_post if oot_post!=None else len(phases)-1][0]]
        im0=ax[0].imshow(data2D[oot_idxs],aspect='auto',extent=extent1)
        fig.colorbar(im0)
        if zoom_in:
            plt.xlim(zoom_in[0],zoom_in[1])
        ax[1].set_xlabel(xlabel)
        ax[0].set_ylabel('Exp. # (OoT)')
        ax[1].set_ylabel('Exp. # (Transit)')
    else:
        fig,ax = plt.subplots(figsize=(21,int(dataset.num_exposures/4)))
        im=ax.imshow(data2D,aspect='auto',extent=[xextent[0],xextent[-1],0,dataset.num_exposures])
        fig.colorbar(im)
        if oot_pre!=None:
            ax.axhline(y=oot_pre+0.5,color='orange',linestyle='dotted',label='Ingress')
        if oot_post!=None:
            ax.axhline(y=oot_post-0.5,color='r',linestyle='dashed',label='Egress')
        if zoom_in:
            plt.xlim(zoom_in[0],zoom_in[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Exp. #')
        plt.legend()
    plt.show()
    return None
# %%
#### PLANET AND STELLAR PARAMS #####
target='wasp76'

vsys = -1.11 #systematic velocity
#vsys = -0.786 #GaiaDR3

t0 = 2460199.99754 #epoch
t0 = 2458080.625664 #epoch
epoch = 2458080.625664 #epoch

per = 1.80988062 #orbital period

t_start = 0.039 #phase marking start of transit

ks = 0.11602

planet = Planet(vsys=vsys, T0=epoch, orbital_period=per)
planet.calc_Kp(mstar=1.458)
planet.transit_start = t_start


#### LOAD DATASET ####

#skip_exp =[0,64]
#skip_exp = list(np.arange(43,59,1))
skip_exp = []
skip_exp = [67,68,69]

start_time = time.time()
dataset = load_espresso_data('../../WASP-82/WASP76_testing_data/',
        target='WASP 76',
        skip_exposures=skip_exp,
        TAC=True,
        mask_tellurics=True,
        cut_off=0.4) 
end_time, end_label = human_time(time.time()-start_time)
print(f"Data loaded in {end_time:.2f} {end_label}.")

orbits=int(round(((int(np.median(dataset.obstimes))-int(epoch))/per),0))
t0=epoch+orbits*per
planet.T0 = t0
# dataset.plot_river()
# plt.show()


id_info = 'espresso_3000K'

### LOAD TEMPLATE ###
temp_dir = '../../WASP-82/templates/'
# temp_dir = '../templates/'

start_time=time.time()
element='Fe'
d = pd.read_csv(temp_dir+target+'_'+element+'_template.csv')
# d = pd.read_csv(temp_dir+'wasp76_Fe_template_1e-5_3000.csv')
template_wl, template = ConvolveToR(d.wavelength.to_numpy()*1e4, d.flux.to_numpy(),100000)
template_wl = pyasl.vactoair2(template_wl)
end_time, end_label = human_time(time.time()-start_time)
print(f"Template loaded in {end_time:.2f} {end_label}.")

pipeline = Pipeline()
pipeline.add_module( WavelengthCutModule(low=3800,high=6910))
# pipeline.add_module( WavelengthCutModule(low=4500,high=6000))
pipeline.add_module( ShiftStellarRestFrameModule2(target=planet,ks=ks,vsys=vsys,correct_vbary=True))
pipeline.add_module( SimpleNormalizationModule())
pipeline.add_module( SigmaClipInterpolateModule(sigma=3))
pipeline.add_module( RemoveBlazeModule(sigma=200))
pipeline.add_module( RemoveHighDeviationPixelsModule(cut_off=2))
# pipeline.add_module( SysRemModule(number_of_modes=2, mode='subtract', savename='cleaned') ) # PCA-ish
#pipeline.add_module( SubtractMasterTemplateModule(target=planet,phase=t_start)) #builds a master spectral template and divides it out of the data (set subtract_oot to FALSE in cross correlation module)
pipeline.add_module( CrossCorrelationModule(template = template, template_wl = template_wl, rv_range=150, drv=0.5, error_weighted=False,mask_rm=False,
    rm_mask_range=[-10,10],subtract_oot=True,target=planet,phase=t_start,savename='CCF'))
pipeline.run(dataset, num_workers=15, per_order=False, plot_stellar_rv=True) 
pipeline.summary()

ccf_map_earth = pipeline.get_results('CCF')

#rvs = ccf_map_earth.rv_grid[0][0]
#rv_idx = np.where( (rvs>-10) & (rvs<10) )

phases = planet.orbital_phase(dataset.obstimes)
vmax = np.percentile(ccf_map_earth.spec, 99.8)

#idx=np.where( (phases>t_start) | (phases<-t_start))[0]
#avg_oot=np.nanmean(ccf_map_earth.spec[idx,0,:],axis=0)
#ccf_map_earth.spec[:,0,:] = (ccf_map_earth.spec[:,0,:]/avg_oot[None,:])-1.0

#Calculate the SNR map (needs t be done before shifting because idk)
Kp_list = np.arange(-50,350, 1.3)
snr_map = make_kp_vsys_map(ccf_map_earth, Kp_list, planet,in_transit=True)

# Save a copy of the ccf so we can get it back without reruning things
old_ccf_map_earth_spec=copy.deepcopy(ccf_map_earth.spec)

plot_exposures(ccf_map_earth.spec[::-1,0,:]*1e6,dataset,planet,xextent=ccf_map_earth.rv_grid[0][0],xlabel='RV [km/s]',separate_transit=False)
plt.show()

# %%
### Apply High-pass Filter ###

# ccf_map_earth.spec=old_ccf_map_earth_spec
hpf = input("Apply a high-pass filter? (y/n)\n")
if hpf=='y':
    print("High-pass filter applied.")
    ccf_map_earth.spec = copy.deepcopy(old_ccf_map_earth_spec)
    ccf_map_earth.spec = hp_filter(ccf_map_earth.spec[:,0,:],dataset.num_exposures,size=100,plot1D=False)
    # plot_exposures(old_ccf_map_earth_spec[:,0,:]*1e6,dataset,planet,xextent=ccf_map_earth.rv_grid[0][0],xlabel='RV [km/s]',separate_transit=False)
    plot_exposures(ccf_map_earth.spec[::-1,0,:]*1e6,dataset,planet,xextent=ccf_map_earth.rv_grid[0][0],xlabel='RV [km/s]',separate_transit=False)
else:
    print("No high-pass filter applied.")
    ccf_map_earth.spec = copy.deepcopy(old_ccf_map_earth_spec)
# ccf_map_earth.spec=highpass_filter(ccf_map_earth.spec)

# ccf_map_earth.spec=old_ccf_map_earth_spec
# ccf_stellar = np.copy(ccf_map_earth.spec[:,0,:]*1e6) #have to make a copy because shifted to the planet frame will change it. This should really all just be put into a another Module

# %%
### Mask out strong RM signal ###

mask_RM = input("Apply RM mask? Input mask width in km/s or 'n' for no mask:\n")
if mask_RM!='n' and mask_RM!='':
    print(f"Applying a mask of {mask_RM} km/s")
    # ccf_map_earth.spec = old_ccf_map_earth_spec
    ccf_stellar = copy.deepcopy(ccf_map_earth.spec[:,0,:]*1e6)
    RM_hw = float(mask_RM)/2
    mask_indices = (ccf_map_earth.rv_grid[0, 0, :] >= -RM_hw) & (ccf_map_earth.rv_grid[0, 0, :] <= RM_hw)
    ccf_stellar[:, mask_indices] = 0
    ccf_map_earth.spec[:, :, mask_indices] = 0
else:
    print(f"No mask applied.")
    # ccf_map_earth.spec = old_ccf_map_earth_spec
    ccf_stellar = copy.deepcopy(ccf_map_earth.spec[:,0,:]*1e6)

### Now shift into the planet frame after masking the RM
pipeline = Pipeline()
pipeline.add_module( ShiftRestFrameModule(target=planet, savename='CCF_planet'))
pipeline.run(ccf_map_earth, num_workers=15, per_order=False)
pipeline.summary()

ccf_planet = pipeline.get_results('CCF_planet')
ccf_planet1 = copy.deepcopy(ccf_planet)
ccf_planet2 = copy.deepcopy(ccf_planet)

#Get 1D CCF for entire transit
pipeline = Pipeline()
pipeline.add_module( CoAddExposures(savename='1D_CCF', weights=planet.in_transit(dataset.obstimes)))
pipeline.run(ccf_planet, num_workers=15, per_order=False)
pipeline.summary()
ccf_1d = pipeline.get_results('1D_CCF')

#Get 1D CCF for first half of transit
pipeline = Pipeline()
pipeline.add_module( CoAddExposures(savename='1D_CCF',weights=planet.half_transit(dataset.obstimes,core=0.02,half='1st half')))
pipeline.run(ccf_planet1, num_workers=15, per_order=False)
pipeline.summary()
ccf_1d_1 = pipeline.get_results('1D_CCF')

#Get 1D CCF for second half of transit
pipeline = Pipeline()
pipeline.add_module( CoAddExposures(savename='1D_CCF', weights=planet.half_transit(dataset.obstimes,core=0.02,half='2nd half')))
pipeline.run(ccf_planet2, num_workers=15, per_order=False)
pipeline.summary()
ccf_1d_2 = pipeline.get_results('1D_CCF')
# %%
### Plots ###

fig2, ax2 = plt.subplots(figsize=(17,12))

idx_out = np.where( np.logical_or(phases<-t_start, phases>t_start))[0]
ccf_std = np.std(np.abs(ccf_stellar[idx_out,:]),axis=1)

y_pos = np.arange(len(ccf_std))
ax2.barh(y_pos,ccf_std,align='center')
ax2.set_yticks(y_pos,labels=dataset.exp_num[idx_out])
ax2.set_xlabel('Exposure Standard Deviation')
ax2.set_ylabel('OoT Exposure #')
for i in np.array([3.0,4.0,5.0])*np.std(ccf_std):
    ax2.axvline(x = np.mean(np.abs(ccf_std)) + i,linestyle='dashed',color='k')


fig3,ax3 = plt.subplots(nrows=2,ncols=2,figsize=(21,21))
fig3.suptitle(element,fontsize=36)
x = np.linspace(np.min(phases),np.max(phases),10)

im3=ax3[0,0].imshow(ccf_stellar, aspect='auto', cmap='gray', origin='lower',
           extent=[ccf_map_earth.rv_grid.min(), ccf_map_earth.rv_grid.max(),min(phases),max(phases)],interpolation='none', vmin=ccf_stellar.min(), vmax=ccf_stellar.max())
ax3[0,0].hlines(xmin=-150,xmax=150,y=(-t_start,t_start),colors='white',linestyles='dashed')
# RM_mask=np.zeros_like(ccf_stellar)
cbar = fig3.colorbar(im3)
cbar.ax.set_ylabel("Stellar Excess Absorption [ppm]",rotation=270,labelpad=20)
# ax3[0,0].imshow(RM_mask, aspect='auto', cmap='gray',extent=[-RM_hw,RM_hw,min(phases),max(phases)],interpolation='none', vmin=ccf_stellar.min(), vmax=ccf_stellar.max())
ax3[0,0].set_xlabel('RV [km/s]')
ax3[0,0].set_ylabel('Phase')
ax3[0,0].plot(planet.Kp*np.sin(x*(2*np.pi)),x,color='white',alpha=0.7)

im4=ax3[0,1].imshow(ccf_planet.spec[:,0,:]*1e6, aspect='auto', cmap='gray', origin='lower',
           extent=[ccf_map_earth.rv_grid.min(), ccf_map_earth.rv_grid.max(),min(phases),max(phases)])
ax3[0,1].set_facecolor("k")
ax3[0,1].hlines(xmin=-150,xmax=150,y=(-t_start,t_start),colors='white',linestyles='dashed')
ax3[0,1].vlines(ymax=np.max(phases),ymin=np.min(phases),x=0.0,colors='white')
cbar = fig3.colorbar(im4)
cbar.ax.set_ylabel("Planet Excess Absorption [ppm]",rotation=270,labelpad=20)
ax3[0,1].set_xlabel(r'RV [km/s]')

im5=ax3[1,0].imshow(snr_map, origin='lower', cmap='gist_heat', aspect='auto', extent=[np.min(ccf_map_earth.rv_grid),np.max(ccf_map_earth.rv_grid), min(Kp_list), max(Kp_list)])
ax3[1,0].hlines(xmin=-150,xmax=150,y=planet.Kp,colors='white',linestyles='dashed')
ax3[1,0].vlines(ymax=np.max(Kp_list),ymin=np.min(Kp_list),x=0.0,colors='white',linestyles='dashed')
ax3[1,0].text(0.65,0.1,f'Max SNR: {int(np.max(snr_map))}',transform=ax3[1,0].transAxes,fontsize=20,color='white')
cbar = fig3.colorbar(im5)
cbar.ax.set_ylabel("SNR",rotation=270,labelpad=15)
ax3[1,0].set_xlabel('V [km/s]')
ax3[1,0].set_ylabel(r'$K_p$ [km/s]')

# ax3[1,1].plot(ccf_map_earth.rv_grid[0,0,:],ccf_1d.spec[0,0,:]*1e6,color='purple')
ax3[1,1].plot(ccf_map_earth.rv_grid[0,0,:],ccf_1d_1.spec[0,0,:]*1e6,color='blue',label='1st Half')
ax3[1,1].plot(ccf_map_earth.rv_grid[0,0,:],ccf_1d_2.spec[0,0,:]*1e6,color='red',label='2nd Half')
xmin_idx = int(np.max(np.where(~np.isnan(ccf_planet.spec[:,0,:]), np.arange(ccf_planet.spec[:,0,:].shape[1]),len(ccf_planet.spec[:,0,:][0])/2), axis=1).min())
xmax_idx = int(np.min(np.where(~np.isnan(ccf_planet.spec[:,0,:]), np.arange(ccf_planet.spec[:,0,:].shape[1]),len(ccf_planet.spec[:,0,:][0])/2), axis=1).max())
xlimit=min(abs(ccf_map_earth.rv_grid[0,0,:][xmin_idx]),abs(ccf_map_earth.rv_grid[0,0,:][xmax_idx]))
ax3[1,1].set_xlim(-xlimit,xlimit)
ymin = min(ccf_1d.spec[0,0,:][min(xmin_idx,xmax_idx):max(xmin_idx,xmax_idx)]*1e6)
ymax = max(ccf_1d.spec[0,0,:][min(xmin_idx,xmax_idx):max(xmin_idx,xmax_idx)]*1e6)
ax3[1,1].set_ylim([1.1*ymin if ymin<0 else 0.9*ymin][0],[1.1*ymax if ymin>0 else 0.9*ymax][0])
offset=abs(np.median((ccf_1d.spec[0,0,:]*1e6)[(ccf_1d.spec[0,0,:]*1e6)<np.percentile(ccf_1d.spec[0,0,:]*1e6,50)]))
coeff, var_matrix = curve_fit(gauss, ccf_map_earth.rv_grid[0,0,:], ccf_1d.spec[0,0,:]*1e6+offset, p0=[max(ccf_1d.spec[0,0,:][min(xmin_idx,xmax_idx):max(xmin_idx,xmax_idx)]*1e6),0,10])
coeff1, var_matrix1 = curve_fit(gauss, ccf_map_earth.rv_grid[0,0,:], ccf_1d_1.spec[0,0,:]*1e6+offset, p0=[max(ccf_1d_1.spec[0,0,:][min(xmin_idx,xmax_idx):max(xmin_idx,xmax_idx)]*1e6),-10,5])
coeff2, var_matrix2 = curve_fit(gauss, ccf_map_earth.rv_grid[0,0,:], ccf_1d_2.spec[0,0,:]*1e6+offset, p0=[max(ccf_1d_2.spec[0,0,:][min(xmin_idx,xmax_idx):max(xmin_idx,xmax_idx)]*1e6),-10,5])
if coeff1[1]<xlimit and coeff1[1]>-xlimit:
    hist_fit1 = gauss(ccf_map_earth.rv_grid[0,0,:], *coeff1)
    ax3[1,1].plot(ccf_map_earth.rv_grid[0,0,:], hist_fit1-offset,color='green')
    #ax3[1,1].vlines(ymin=0,ymax=1,x=0.0,colors='k',linestyles='dashed',transform=ax3[1,1].get_yaxis_transform())
    # ax3[1,1].axvline(x=coeff[1],color='k',linestyle='dashed',label=f"{coeff[1]:.2f} km/s")
else:
    print(f"Fit 1 not plotted. Unrealistic gaussian peaked at {coeff1[1]:.2f} km/s")
if coeff2[1]<xlimit and coeff2[1]>-xlimit:
    hist_fit2 = gauss(ccf_map_earth.rv_grid[0,0,:], *coeff2)
    ax3[1,1].plot(ccf_map_earth.rv_grid[0,0,:], hist_fit2-offset,color='orange')
    #ax3[1,1].vlines(ymin=0,ymax=1,x=0.0,colors='k',linestyles='dashed',transform=ax3[1,1].get_yaxis_transform())
    # ax3[1,1].axvline(x=coeff[1],color='k',linestyle='dashed',label=f"{coeff[1]:.2f} km/s")
else:
    print(f"Fit 2 not plotted. Unrealistic gaussian peaked at {coeff2[1]:.2f} km/s")
ax3[1,1].set_xlabel('Radial Velocity [km/s]')
ax3[1,1].set_ylabel('Co-Added Excess Absorption [ppm]',rotation=270,labelpad=15)
ax3[1,1].yaxis.set_label_position('right')
ax3[1,1].tick_params(axis='y', labelright=True, right=True)
ax3[1,1].legend()
# print(ax3[1,1].get_ylim())
plt.tight_layout()
print(f"Max SNR: {np.max(snr_map)}")
#plt.savefig(f"./wasp76_{id_info}_Fe.pdf",format='pdf')
plt.show()
print(f"Gaussian fit for total transit: {coeff[0]:.0f} ppm at {coeff[1]:.2f} km/s")
print(f"Gaussian fit for 1st half of transit: {coeff1[0]:.0f} ppm at {coeff1[1]:.2f} km/s")
print(f"Gaussian fit for 2nd half of transit: {coeff2[0]:.0f} ppm at {coeff2[1]:.2f} km/s")
# %%
# %%
response = input("Save CCF's? (y/n)\n")

if response =='y':
    save_dict = {'ccf_stellar':ccf_stellar,'ccf_planet':ccf_planet.spec,'phase':phases,'rv':ccf_planet.wavelengths[0,0,:],'snr_map':snr_map}

    with open(f"../../results/wasp76_{id_info}_fe_ccf_planet.pkl","wb") as f:
        pickle.dump(save_dict,f)

# %%
dataset.num_exposures
planet.orbital_phase(ccf_map_earth.obstimes)
ccf_map_earth.info()
# %%
