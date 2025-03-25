#------ manually fix import for now
import sys
import pathlib

parent_folder = pathlib.Path(__file__).absolute().parents[1]
sys.path.append(str(parent_folder))
#------

import numpy as np
import astropy.units as u
from redexo import *
import matplotlib.pyplot as plt

planet = Planet(Kp=196, vsys=-1.1, T0=2458080.626165, orbital_period=1.809886)
planet.transit_start = 0.043

dataset = load_carmenes_data(folder='/users/ricolandman/Documents/Research/OH_exoplanets/Data/Wasp76_NIR/',\
                            spectral_orders=[23,24,25,26,27])
template_wl, template = np.load('/users/ricolandman/Documents/Research/OH_exoplanets/Data/models/Wasp76_OH_equilibrium_T=2150K_prepared.npy')

pipeline = Pipeline()

#pipeline.add_module( InjectSignalModule(template=template, template_wl=template_wl, target=planet, savename='injection') )
pipeline.add_module( FlagAbsorptionEmissionModule(flux_lower_limit=0.2, flux_upper_limit=1.1, relative_to_continuum=True) )
pipeline.add_module( PolynomialContinuumRemovalModule(poly_order=3))
pipeline.add_module( OutlierFlaggingModule(sigma=5) )

pipeline.add_module( SysRemModule(number_of_modes=12, mode='subtract', savename='cleaned') )

pipeline.add_module( CrossCorrelationModule(template = template, template_wl = template_wl, rv_range=250, drv=1.3, error_weighted=True))
pipeline.add_module( CoAddOrders(savename='co_added', weights=None))
pipeline.add_module( ShiftRestFrameModule(target=planet, savename='CCF_map_planet'))
pipeline.add_module( CoAddExposures(savename='1D_CCF', weights=planet.in_transit(dataset.obstimes)))

pipeline.run(dataset, num_workers=5, per_order=True) 
pipeline.summary()

ccf_map_earth = pipeline.get_results('co_added')
ccf_map_planet = pipeline.get_results('CCF_map_planet')

#Get Kp-vsys map
Kp_list = np.arange(-50,350, 1.3)
snr_map = make_kp_vsys_map(ccf_map_earth, Kp_list, planet)
plt.imshow(snr_map, origin='lower', cmap='gist_heat', aspect='auto', extent=[np.min(ccf_map_earth.rv_grid), np.max(ccf_map_earth.rv_grid), min(Kp_list), max(Kp_list)])
plt.ylabel(r'$K_p$ [km/s]', fontsize=14)
plt.xlabel(r'$v_{sys}$ [km/s]',fontsize=14)
plt.colorbar(label='SNR')
plt.tight_layout()
print('SNR:', np.max(snr_map))

plt.figure()
plt.imshow(ccf_map_earth.spec[:,0,:], aspect='auto', cmap='inferno', origin='lower')
plt.colorbar()
plt.show()
