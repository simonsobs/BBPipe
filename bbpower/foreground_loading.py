import numpy as np

from fgbuster.component_model import CMB, Dust, Synchrotron, AnalyticComponent, FreeFree 

hplanck = 6.626068e-34
kboltz = 1.3806503e-23

class FGModel:
    def __init__(self, config):
        self.load_foregrounds(config)
        return 

    def load_foregrounds(self, config):
        """
        Loads the foreground models and prepares the unit conversions to K_CMB units. 
        """
        self.components = {}
        for key in config['fg_model']:
            self.components[key] = {}

        for key, component in config['fg_model'].items(): 
            nu0 = component['parameters']['nu0']
            sed_fnc = get_fgbuster_sed(component['sed'])
            self.components[key]['sed'] = sed_fnc(**component['parameters'], units='K_RJ')
            self.components[key]['cmb_n0_norm'] = CMB('K_RJ').eval(nu0) * nu0**2
            self.components[key]['nu0'] = nu0
            self.components[key]['spectrum_params'] = component['spectrum']
        return 
    

class FGParameters: 
    def __init__(self, config):
        self.define_parameters(config)
        return 

    def define_parameters(self, config):
        self.param_init = []
        self.param_index = {}
        self.amp_index = {} 
        self.priors = {}
        self.param_index['r'] = 0
        self.param_init.append(0.02)
        pindx = 1
        for key, component in config['fg_model'].items():
            for param, prior in component['priors'].items():
                self.param_index[param] = pindx
                if prior[0].lower() == 'tophat':
                    init_indx = 1
                elif prior[0].lower() == 'gaussian':
                    init_indx = 0
                self.param_init.append(prior[1][init_indx])
                self.priors[param] = prior
                if 'amp' in param:
                    self.amp_index[key] = pindx
                pindx += 1
        return

def get_fgbuster_sed(sed_name):
    sed_fncs = {
        'Dust': Dust, 
        'ModifiedBlackBody': Dust, 
        'Synchrotron': Synchrotron,
        'PowerLaw': Synchrotron,
        'FreeFree': FreeFree,
        'AnalyticComponent': AnalyticComponent
        }
    try:
        return sed_fncs[sed_name] 
    except KeyError:
        print("Foreground named %s is not a valid FGBuster foreground name" %(sed_name))
    return 

def normed_plaw(ell, alpha):
    ell0 = 80.
    return (ell/ell0)**alpha 



