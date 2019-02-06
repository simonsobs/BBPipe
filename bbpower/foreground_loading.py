import numpy as np

from fgbuster.component_model import CMB, Dust, Synchrotron, AnalyticComponent, AME, FreeFree 

class FGModel:
    def __init__(self, config):
        self.load_foregrounds(config)
        return 

    def load_foregrounds(self, config):
        """
        Loads the foreground models and prepares the unit conversions to K_CMB units. 
        """
        self.components = {}
        for key, component in config['fg_model']:
            self.components[key] = {}

        for key, component in config['fg_model'].items(): 
            nu0 = component['parameters']['nu0']
            sed_fnc = get_fgbuster_sed(component.sed)
            self.components[key]['sed'] = sed_fnc(**component['parameters'], units='K_RJ')
            self.components[key]['cmb_n0_norm'] = CMB('K_RJ').eval(nu0) * nu0**2
            self.components[key]['nu0'] = nu0
        return 

    # TODO: define evaluate SED and evaluate power spectrum methods. 
    
    def evaluate_sed(self, nus): 
        # do cool stuff. 
        for key, component in self.fg_model.components: 
            conv_rj = (nus / component['nu0'])**2

            sed_params = [] 
            for param in self.fg_sed_model[component].params:
                pindx = self.param_index[param]
                fg_comp_params.append(params[pindx])

            fg_sed_eval = self.fg_sed_model[component].eval(nus, *fg_comp_params) * conv_rj

        return sed 
    
    def evaluate_power_spectrum(self, ells): 
        # do more cool stuff 
        return power_spectrum 
    

class FGParameters: 
    def __init__(self):
        self.define_parameters(config)
        return 

    def define_parameters(self, config):
        self.parameters = {}


        self.param_init = []
        self.param_index = {}
        self.amp_index = {} 
        self.param_index['r'] = 0
        self.param_init.append(0)
        pindx = 1
        for key, component in config['fg_model'].items():
            for param in component['priors']:
                self.component['key']['sed_param_index'][param] = pindx
                self.param_init.append(self.fg_model[component]['priors'][param][1])
                if 'amp' in param:
                    self.amp_index[component] = pindx
                pindx += 1
                # TODO: need something for cross correlation params as well 
        return

def get_fgbuster_sed(sed_name):
    sed_fncs = {
        'Dust': Dust, 
        'ModifiedBlackBody': Dust, 
        'Synchrotron': Synchrotron,
        'PowerLaw': Synchrotron,
        'AME': AME, 
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



