__all__ = ['Pipeline']

import numpy as np
from astropy.io import fits
from pathos.multiprocessing import ProcessingPool as Pool
from .dataset import Dataset, CCF_Dataset
import time


class Pipeline(object):
    def __init__(self, in_memory=True):
        self.modules = []
        self.database = {}

    def add_module(self, module):
        self.modules.append(module)

    def run(self, dataset, per_order=True, num_workers=None, plot_stellar_rv=False, debug=False):
        processed_dataset = dataset
        self.run_times = []
        for module in self.modules:
            t = time.time()
            if (per_order or module.per_order) and module.per_order_possible:
                if module.__class__.__name__ == "CrossCorrelationModule":
                    if num_workers is not None:
                        with Pool(num_workers) as p:
                            results = p.map(lambda order: module.process(processed_dataset.get_order(order), plot_stellar_rv=plot_stellar_rv, debug=debug), range(processed_dataset.num_orders))
                    else:
                        results = [module.process(processed_dataset.get_order(order), plot_stellar_rv=plot_stellar_rv, debug=debug) for order in range(processed_dataset.num_orders)]
                else:
                    if num_workers is not None:
                        with Pool(num_workers) as p:
                            results = p.map(lambda order: module.process(processed_dataset.get_order(order), debug=debug), range(processed_dataset.num_orders))
                    else:
                        results = [module.process(processed_dataset.get_order(order), debug=debug) for order in range(processed_dataset.num_orders)]

                #Check if resulting dataset still matches our previous dataset
                #If not, make sure they get compatible
                if isinstance(results[0], CCF_Dataset) and not isinstance(processed_dataset, CCF_Dataset):
                    empty_spec = np.zeros((results[0].spec.shape[0], processed_dataset.num_orders, results[0].spec.shape[-1]))
                    processed_dataset = CCF_Dataset(empty_spec,rv_grid=empty_spec.copy(),\
                             vbar=processed_dataset.vbar, obstimes=processed_dataset.obstimes, **dataset.header_info)
                elif not results[0].spec.shape==processed_dataset.spec.shape:
                    processed_dataset.make_clean(shape=(results[0].spec.shape[0], processed_dataset.num_orders, results[0].spec.shape[-1]))

                #Write results to dataset
                for order, res in enumerate(results):
                    processed_dataset.set_results(res, order=order)
            else:
                if module.__class__.__name__ == "CrossCorrelationModule":
                    processed_dataset = module.process(processed_dataset, plot_stellar_rv=plot_stellar_rv, debug=debug)
                else:
                    processed_dataset = module.process(processed_dataset, debug=debug)
            if not module.savename is None:
                self.database[module.savename] = processed_dataset.copy()
            self.run_times.append(time.time()-t)
            #print(module, processed_dataset, processed_dataset.spec.shape)

    def get_results(self, name):
        return self.database[name]

    def write_results(self, filepath):
        self.hdulist = fits.HDUList()
        self.hdulist.append(fits.PrimaryHDU())
        for savename, dataset in self.database.items():
            self.hdulist.append(fits.ImageHDU(dataset.spec, name=savename+"_spec"))
            self.hdulist.append(fits.ImageHDU(dataset.wavelengths, name=savename+"_wavelengths"))
            self.hdulist.append(fits.ImageHDU(dataset.errors, name=savename+"_errors"))
        self.hdulist.writeto(filepath)

    def summary(self):
        print('----------Summary--------')
        for i, module in enumerate(self.modules):
            print('Running {0} took {1:.2f} seconds'.format(module, self.run_times[i]))
        print('--> Total time: {0:.2f} seconds'.format(np.sum(np.array(self.run_times))))
        print('------------------------')
