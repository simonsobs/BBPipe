import numpy as np


class ParameterManager(object):

    def _add_parameter(self, p_name, p):
        # If fixed parameter, just add its name and value
        if p[1] == 'fixed':
            self.p_fixed.append((p_name, p[2][0]))
            return  # Then move on

        # Otherwise it's free
        # Check for duplicate names
        if p_name in self.p_free_names:
            raise KeyError("You have two parameters with the same name")
        # Add name and prior to list
        self.p_free_names.append(p_name)
        self.p_free_priors.append(p)
        # Add fiducial value to initial vector
        if p[1] == 'tophat':
            p0 = p[2][1]
        elif p[1] == 'Gaussian':
            p0 = p[2][0]
        else:
            raise ValueError("Unknown prior type %s" % p[1])
        self.p0.append(p0)

    def _add_parameters(self, params):
        for p_name in sorted(params.keys()):
            p = params[p_name]
            self._add_parameter(p_name, p)

    def __init__(self,config):
        self.p_free_names = []
        self.p_free_priors = []
        self.p_fixed = []
        self.p0 = []

        # CMB parameters
        d = config.get('cmb_model')
        if d:
            self._add_parameters(d['params'])

        # Loop through FG components
        for c_name in sorted(config['fg_model'].keys()):
            c = config['fg_model'][c_name]
            for tag in ['sed_parameters', 'cl_parameters', 'cross']:
                d = c.get(tag)
                if d:
                    self._add_parameters(d)

    def build_params(self, par):
        params = dict(self.p_fixed)
        params.update(dict(zip(self.p_free_names, par)))
        return params

    def lnprior(self, par):
        lnp = 0
        for p, pr in zip(par, self.p_free_priors):
            if pr[1] == 'Gaussian': #Gaussian prior
                lnp += -0.5 * ((p - pr[2][0])/pr[2][1])**2
            else: #Only other option is top-hat
                if not(float(pr[2][0]) <= p <= float(pr[2][2])):
                    return -np.inf
        return lnp
