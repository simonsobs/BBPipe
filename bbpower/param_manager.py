import numpy as np


class ParameterManager(object):

    def _add_parameter(self, p_name, p):
        if p[1] == 'fixed':
            self.p_fixed.append((p_name, float(p[2][0])))
            return 

        if p_name in self.p_free_names:
            raise KeyError("You have two parameters with the same name")
        self.p_free_names.append(p_name)
        self.p_free_priors.append(p)
        if np.char.lower(p[1]) == 'tophat':
            p0 = float(p[2][1])
        elif np.char.lower(p[1]) == 'gaussian':
            p0 = float(p[2][0])
        else:
            raise ValueError("Unknown prior type %s" % p[1])
        self.p0.append(p0)
        return 

    def _add_parameters(self, params):
        for p_name in sorted(params.keys()):
            p = params[p_name]
            self._add_parameter(p_name, p)
        return 

    def __init__(self,config):
        self.p_free_names = []
        self.p_free_priors = []
        self.p_fixed = []
        self.p0 = []

        d = config.get('cmb_model')
        if d:
            self._add_parameters(d['params'])

        for c_name in sorted(config['fg_model'].keys()):
            c = config['fg_model'][c_name]
            for tag in ['sed_parameters', 'cross', 'decorr']:
                d = c.get(tag)
                if d:
                    self._add_parameters(d)
            dc = c.get('cl_parameters')
            if dc:
                for cl_name,d in dc.items():
                    p1,p2=cl_name
                    # Add parameters only if we're using both polarization channels
                    if (p1 in config['pol_channels']) and (p2 in config['pol_channels']):
                        self._add_parameters(d)

        if 'systematics' in config.keys():
            cnf_sys = config['systematics']
            if 'bandpasses' in cnf_sys.keys():
                cnf_bps = cnf_sys['bandpasses']
                i_bps = 1
                while 'bandpass_%d' % i_bps in cnf_bps:
                    if cnf_bps['bandpass_%d' % i_bps].get('parameters'):
                        self._add_parameters(cnf_bps['bandpass_%d' % i_bps]['parameters'])
                    i_bps += 1

        self.p0 = np.array(self.p0)
        return 

    def build_params(self, par):
        params = dict(self.p_fixed)
        params.update(dict(zip(self.p_free_names, par)))
        return params

    def lnprior(self, par):
        lnp = 0
        for p, pr in zip(par, self.p_free_priors):
            if np.char.lower(pr[1]) == 'gaussian': 
                lnp += -0.5 * ((p - pr[2][0])/pr[2][1])**2
            else: #Only other option is top-hat
                if not(float(pr[2][0]) <= p <= float(pr[2][2])):
                    return -np.inf
        return lnp
