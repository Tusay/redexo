# %%
#!/usr/bin/env python3

#Default Modules:
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
import pickle
from PyAstronomy import pyasl
#Other Modules:
sys.path.append('../redexo')
##### Author: Alex Polanski #####

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
# %%
#### PLANET AND STELLAR PARAMS #####

vsys = -1.11 #systematic velocity
#vsys = -0.786 #GaiaDR3

t0 = 2460199.99754 #epoch
t0 = 2458080.625664 #epoch

per = 1.80988062 #orbital period

t_start = 0.039 #phase marking start of transit

ks = 0.11602

planet = Planet(vsys=vsys, T0=t0, orbital_period=per)
planet.calc_Kp(mstar=1.458)
planet.transit_start = t_start


#### LOAD DATASET ####

#skip_exp =[0,64]
#skip_exp = list(np.arange(43,59,1))
skip_exp = []
skip_exp = [67,68,69]

start_time = time.time()
dataset = load_espresso_data('../../../WASP-82/WASP76_testing_data/',
        target='WASP 76',
        skip_exposures=skip_exp,
        TAC=True,
        mask_tellurics=True,
        cut_off=0.4) 
end_time, end_label = human_time(time.time()-start_time)
print(f"Data loaded in {end_time:.2f} {end_label}.")
# dataset = load_kpf_data('../data/kpf_wasp76/',
#         star_name='WASP 76',
#         skip_exposures=skip_exp,
#         TAC=True,
#         mask_tellurics=True,
#         cut_off=0.4) 

dataset.plot_river()
plt.show()


id_info = 'espresso_3000K'

### LOAD TEMPLATE ###
temp_dir = '../../../WASP-82/templates/'
# temp_dir = '../templates/'

start_time=time.time()
d = pd.read_csv(temp_dir+'wasp76_Fe_template.csv')
# d = pd.read_csv(temp_dir+'wasp76_Fe_template_1e-5_3000.csv')
template_wl, template = ConvolveToR(d.wavelength.to_numpy()*1e4, d.flux.to_numpy(),100000)
template_wl = pyasl.vactoair2(template_wl)
end_time, end_label = human_time(time.time()-start_time)
print(f"Template loaded in {end_time:.2f} {end_label}.")

pipeline = Pipeline()
pipeline.add_module( WavelengthCutModule(low=3800,high=7800))
# pipeline.add_module( WavelengthCutModule(low=4500,high=6000))
pipeline.add_module( ShiftStellarRestFrameModule2(target=planet,ks=ks,vsys=vsys,correct_vbary=True))
pipeline.add_module( SimpleNormalizationModule())
pipeline.add_module( SigmaClipInterpolateModule(sigma=3))
pipeline.add_module( RemoveBlazeModule(sigma=200))
pipeline.add_module( RemoveHighDeviationPixelsModule(cut_off=2))
#pipeline.add_module( SubtractMasterTemplateModule(target=planet,phase=t_start)) #builds a master spectral template and divides it out of the data (set subtract_oot to FALSE in cross correlation module)
pipeline.add_module( CrossCorrelationModule(template = template, template_wl = template_wl, rv_range=150, drv=0.5, error_weighted=False,
    mask_rm=False,rm_mask_range=[-10,10],subtract_oot=False,target=planet,phase=t_start,savename='CCF'))
pipeline.run(dataset, num_workers=15, per_order=False) 
pipeline.summary()

fig,ax = plt.subplots(figsize=(21,int(dataset.num_exposures/4)))
ccf_map_earth = pipeline.get_results('CCF')
ax.imshow(ccf_map_earth.spec[:,0,:]*1e6,extent=[ccf_map_earth.rv_grid[0][0][0],ccf_map_earth.rv_grid[0][0][-1],0,dataset.num_exposures])
ax.set_xlabel('RV [km/s]')
ax.set_ylabel('Exp. #')
plt.show()

start_time=time.time()
dataset = load_espresso_data('../../../WASP-82/WASP76_testing_data/',
        target='WASP 76',
        skip_exposures=skip_exp,
        TAC=True,
        mask_tellurics=True,
        cut_off=0.4) 
end_time, end_label = human_time(time.time()-start_time)
print(f"Data loaded in {end_time:.2f} {end_label}.")

pipeline = Pipeline()
pipeline.add_module( WavelengthCutModule(low=3800,high=7800))
# pipeline.add_module( WavelengthCutModule(low=4500,high=6000))
pipeline.add_module( ShiftStellarRestFrameModule2(target=planet,ks=ks,vsys=vsys,correct_vbary=True))
pipeline.add_module( SimpleNormalizationModule())
pipeline.add_module( SigmaClipInterpolateModule(sigma=3))
pipeline.add_module( RemoveBlazeModule(sigma=200))
pipeline.add_module( RemoveHighDeviationPixelsModule(cut_off=2))
#pipeline.add_module( SubtractMasterTemplateModule(target=planet,phase=t_start)) #builds a master spectral template and divides it out of the data (set subtract_oot to FALSE in cross correlation module)
pipeline.add_module( CrossCorrelationModule(template = template, template_wl = template_wl, rv_range=150, drv=0.5, error_weighted=False,
    mask_rm=False,rm_mask_range=[-10,10],subtract_oot=True,target=planet,phase=t_start,savename='CCF'))

pipeline.run(dataset, num_workers=15, per_order=False) 
pipeline.summary()

fig,ax = plt.subplots(figsize=(21,int(dataset.num_exposures/4)))
im=ax.imshow(dataset.spec.reshape(dataset.num_exposures,-1),aspect='auto',extent=[dataset.wavelengths[0][0][0],dataset.wavelengths[0][0][-1],0,dataset.num_exposures])
fig.colorbar(im)
ax.set_xlabel(r'Wavelength [$\AA$]')
ax.set_ylabel('Exposure Number')
plt.show()

fig,ax = plt.subplots(figsize=(21,int(dataset.num_exposures/4)))
ccf_map_earth = pipeline.get_results('CCF')
ax.imshow(ccf_map_earth.spec[:,0,:]*1e6,extent=[ccf_map_earth.rv_grid[0][0][0],ccf_map_earth.rv_grid[0][0][-1],0,dataset.num_exposures])
ax.set_xlabel('RV [km/s]')
ax.set_ylabel('Exp. #')
plt.show()

ccf_map_earth = pipeline.get_results('CCF')

#rvs = ccf_map_earth.rv_grid[0][0]
#rv_idx = np.where( (rvs>-10) & (rvs<10) )

phases = planet.orbital_phase(dataset.obstimes)
vmax = np.percentile(ccf_map_earth.spec, 99.8)

#idx=np.where( (phases>t_start) | (phases<-t_start))[0]
#avg_oot=np.nanmean(ccf_map_earth.spec[idx,0,:],axis=0)
#ccf_map_earth.spec[:,0,:] = (ccf_map_earth.spec[:,0,:]/avg_oot[None,:])-1.0


ccf_stellar = np.copy(ccf_map_earth.spec[:,0,:]*1e6) #have to make a copy because shifted to the planet frame will change it. This should really all just be put into a another Module

#Calculate the SNR map (needs t be done before shifting because idk)
Kp_list = np.arange(-50,350, 1.3)
snr_map = make_kp_vsys_map(ccf_map_earth, Kp_list, planet,in_transit=True)


#Now shift into the planet frame after masking the RM
pipeline = Pipeline()
pipeline.add_module( ShiftRestFrameModule(target=planet, savename='CCF_planet'))
pipeline.run(ccf_map_earth, num_workers=15, per_order=False)
pipeline.summary()

ccf_planet = pipeline.get_results('CCF_planet')

#Get 1D CCF
pipeline = Pipeline()
print(planet.in_transit(dataset.obstimes))
pipeline.add_module( CoAddExposures(savename='1D_CCF', weights=planet.in_transit(dataset.obstimes)))
pipeline.run(ccf_planet, num_workers=15, per_order=False)
pipeline.summary()

ccf_1d = pipeline.get_results('1D_CCF')
# %%
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

x = np.linspace(np.min(phases),np.max(phases),10)

im3=ax3[0,0].imshow(ccf_stellar, aspect='auto', cmap='gray', origin='lower',
           extent=[ccf_map_earth.rv_grid.min(), ccf_map_earth.rv_grid.max(),min(phases),max(phases)],interpolation='none')
ax3[0,0].hlines(xmin=-150,xmax=150,y=(-t_start,t_start),colors='white',linestyles='dashed')
ax3[0,0].set_xlabel('RV [km/s]')
ax3[0,0].set_ylabel('Phase')
ax3[0,0].plot(planet.Kp*np.sin(x*(2*np.pi)),x,color='white',alpha=0.7)

im4=ax3[0,1].imshow(ccf_planet.spec[:,0,:]*1e6, aspect='auto', cmap='gray', origin='lower',
           extent=[ccf_map_earth.rv_grid.min(), ccf_map_earth.rv_grid.max(),min(phases),max(phases)])
ax3[0,1].set_facecolor("k")
ax3[0,1].hlines(xmin=-150,xmax=150,y=(-t_start,t_start),colors='white',linestyles='dashed')
ax3[0,1].vlines(ymax=np.max(phases),ymin=np.min(phases),x=0.0,colors='white')
cbar = fig3.colorbar(im4)
cbar.ax.set_ylabel("Excess Absorption [ppm]",rotation=270,labelpad=20)
ax3[0,1].set_xlabel(r'RV [km/s]')

im5=ax3[1,0].imshow(snr_map, origin='lower', cmap='gist_heat', aspect='auto', extent=[np.min(ccf_map_earth.rv_grid),np.max(ccf_map_earth.rv_grid), min(Kp_list), max(Kp_list)])
ax3[1,0].hlines(xmin=-150,xmax=150,y=planet.Kp,colors='white',linestyles='dashed')
ax3[1,0].vlines(ymax=np.max(Kp_list),ymin=np.min(Kp_list),x=0.0,colors='white',linestyles='dashed')
ax3[1,0].text(0.65,0.1,f'Max SNR: {int(np.max(snr_map))}',transform=ax3[1,0].transAxes,fontsize=20,color='white')
cbar = fig3.colorbar(im5)
cbar.ax.set_ylabel("SNR",rotation=270,labelpad=15)
ax3[1,0].set_xlabel('V [km/s]')
ax3[1,0].set_ylabel(r'$K_p$ [km/s]')

ax3[1,1].plot(ccf_map_earth.rv_grid[0,0,:],ccf_1d.spec[0,0,:]*1e6,color='blue')
ax3[1,1].set_xlim(-60,60)
offset=abs(np.median((ccf_1d.spec[0,0,:]*1e6)[(ccf_1d.spec[0,0,:]*1e6)<np.percentile(ccf_1d.spec[0,0,:]*1e6,50)]))
coeff, var_matrix = curve_fit(gauss, ccf_map_earth.rv_grid[0,0,:], ccf_1d.spec[0,0,:]*1e6+offset, p0=[800,-10,20])
hist_fit = gauss(ccf_map_earth.rv_grid[0,0,:], *coeff)
ax3[1,1].plot(ccf_map_earth.rv_grid[0,0,:], hist_fit-offset,color='green')
#ax3[1,1].vlines(ymin=0,ymax=1,x=0.0,colors='k',linestyles='dashed',transform=ax3[1,1].get_yaxis_transform())
ax3[1,1].axvline(x=coeff[1],color='k',linestyle='dashed',label=f"{coeff[1]:.2f} km/s")
ax3[1,1].set_xlabel('Radial Velocity [km/s]')
ax3[1,1].set_ylabel('Co-Added Excess Absorption [ppm]')
ax3[1,1].legend()

plt.tight_layout()
print(f"Max SNR: {np.max(snr_map)}")
#plt.savefig(f"./wasp76_{id_info}_Fe.pdf",format='pdf')
plt.show()
# %%
# %%
response = input("Save CCF's? (y/n)\n")

if response =='y':
    save_dict = {'ccf_stellar':ccf_stellar,'ccf_planet':ccf_planet.spec,'phase':phases,'rv':ccf_planet.wavelengths[0,0,:],'snr_map':snr_map}

    with open(f"../../results/wasp76_{id_info}_fe_ccf_planet.pkl","wb") as f:
        pickle.dump(save_dict,f)
