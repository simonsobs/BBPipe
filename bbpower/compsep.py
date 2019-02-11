import numpy as np

from bbpipe import PipelineStage
from .types import DummyFile, YamlFile
from .foreground_loading import FGModel, FGParameters, normed_plaw
from fgbuster.component_model import CMB 
from sacc.sacc import SACC

import emcee

class BBCompSep(PipelineStage):
    """
    Component separation stage
    This stage does harmonic domain foreground cleaning (e.g. BICEP).
    The foreground model parameters are defined in the config.yml file. 
    """
    name = "BBCompSep"
    inputs = [('sacc_file', SACC)]
    outputs = [('param_chains', DummyFile)]

    def setup_compsep(self):
        """
        Pre-load the data, CMB BB power spectrum, and foreground models.
        """
        self.parse_sacc_file()
        self.load_cmb()
        self.fg_model = FGModel(self.config)
        self.parameters = FGParameters(self.config)
        return

    def parse_sacc_file(self):
        """
        Reads the data in the sacc file included the power spectra, bandpasses, and window functions. 
        """
        self.s = SACC.loadFromHDF(self.get_input('sacc_file'))
        self.order = self.s.sortTracers()

        self.bpasses = []
        for t in self.s.tracers :
            nu = t.z
            dnu = np.zeros_like(nu);
            dnu[1:-1] = 0.5 * (nu[2:] - nu[:-2])
            dnu[0] = nu[1] - nu[0]
            dnu[-1] = nu[-1] - nu[-2]
            bnu = t.Nz
            self.bpasses.append([nu, dnu, bnu])

        self.bpw_l = self.s.binning.windows[0].ls
        self.data = self.s.mean.vector
        self.covar = self.s.precision.getCovarianceMatrix()

        self.n_tracers = np.arange(12) 
        self.indx = []
        for t1,t2,typ,ells,ndx in self.order:
            if typ == b'BB':
                self.indx += list(ndx)
        self.bbdata = self.data[self.indx]
        self.bbcovar = self.covar[self.indx][:, self.indx]
        self.invcov = np.linalg.solve(self.bbcovar, np.identity(len(self.bbcovar)))
        return

    def load_cmb(self):
        """
        Loads the CMB BB spectrum as defined in the config file. 
        """
        cmb_lensingfile = np.loadtxt(self.config['cmb_files'][0])
        cmb_bbfile = np.loadtxt(self.config['cmb_files'][1])
        
        self.cmb_ells = cmb_bbfile[:, 0]
        mask = self.cmb_ells <= self.bpw_l.max()
        self.cmb_ells = self.cmb_ells[mask] 
        self.cmb_bbr = cmb_bbfile[:, 3][mask]
        self.cmb_bblensing = cmb_lensingfile[:, 3][mask]
        self.get_cmb_norms()
        return

    def get_cmb_norms(self):
        """
        Evaulates the CMB unit conversion over the bandpasses. 
        """
        cmb_norms = [] 
        for tn in self.n_tracers:
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            #bpass_integration = dnu*bpass
            bpass_integration = bpass * nus**2 * dnu

            cmb_thermo_units = CMB('K_RJ').eval(nus) #* nus**2 
            cmb_norms.append(np.dot(bpass_integration, cmb_thermo_units))
        self.cmb_norm = np.asarray(cmb_norms)
        return
    
    def integrate_seds(self, params):
        fg_scaling = {}
        for key in self.fg_model.components:
            fg_scaling[key] = []

        meannu = {}
        for tn in self.n_tracers:
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            bpass_integration = bpass * nus**2 * dnu
            meannu[tn] = np.sum(dnu * nus * bpass) / np.sum(dnu * bpass)

            for key, component in self.fg_model.components.items(): 
                #conv_rj = (nus / component['nu0'])**2

                sed_params = [] 
                for param in component['sed'].params:
                    pindx = self.parameters.param_index[param]
                    sed_params.append(params[pindx])
                
                #fg_units = 1. / self.cmb_norm[tn]
                fg_units = component['cmb_n0_norm'] / self.cmb_norm[tn]
                #fg_sed_eval = component['sed'].eval(nus, *sed_params) * conv_rj
                fg_sed_eval = component['sed'].eval(nus, *sed_params) 
                fg_sed_int = np.dot(fg_sed_eval, bpass_integration) * fg_units
                fg_scaling[key].append(fg_sed_int)

        return fg_scaling, meannu
    
    def evaluate_power_spectra(self, params):
        fg_pspectra = {}
        for key, component in self.fg_model.components.items():
            pspec_params = []
            # TODO: generalize for different power spectrum models
            # should look like:
            # for param in power_spectrum_model: get param index (same as the SEDs)
            for param in component['spectrum_params']:
                pindx = self.parameters.param_index[param]
                pspec_params.append(params[pindx])
            fg_pspectra[key] = normed_plaw(self.bpw_l, *pspec_params)
        return fg_pspectra
    
    def model(self, params):
        """
        Defines the total model and integrates over the bandpasses and windows. 
        """
        cmb_bmodes = params[0] * self.cmb_bbr + self.cmb_bblensing
        fg_scaling = self.integrate_seds(params)
        fg_p_spectra = self.evaluate_power_spectra(params)
        
        cls_array_list = [] 
        for t1,t2,typ,ells,ndx in self.order:
            if typ == b'BB':
                windows = self.s.binning.windows[ndx]
                
                model = cmb_bmodes
                for component in self.fg_model.components:
                    sed_power_scaling = fg_scaling[component][t1] * fg_scaling[component][t2]
                    p_amp = params[self.parameters.amp_index[component]]
                    model += p_amp * sed_power_scaling * fg_p_spectra[component]
                
                    config_component = self.config['fg_model'][component]
                    if 'cross' in config_component.keys():
                        epsilon = config_component['cross']['param']
                        epsilon_index = self.parameters.param_index[epsilon]
                        cross_name = config_component['cross']['name']

                        cross_scaling = fg_scaling[component][t1] * fg_scaling[cross_name][t2] + \
                                        fg_scaling[cross_name][t1] * fg_scaling[component][t2]
                        cross_spectrum = np.sqrt(fg_p_spectra[component] * fg_p_spectra[cross_name])
                        cross_amp = np.sqrt(p_amp * params[self.parameters.amp_index[cross_name]])

                        model += params[epsilon_index] * cross_amp * cross_scaling * cross_spectrum

                model = np.asarray([np.dot(w.w, model) for w in windows])
                cls_array_list.append(model)
        
        return np.asarray(cls_array_list).reshape(len(self.indx), ) 

    def lnpriors(self, params):
        """
        Assign priors for emcee. 
        """
        total_prior = 0
        if params[0] < 0:
            return -np.inf
        
        for key, prior in self.parameters.priors.items():
            pval = params[self.parameters.param_index[key]]

            if prior[0].lower() == 'gaussian':
                mu = prior[1][0]
                sigma = prior[1][1]
                total_prior += -0.5 * (pval - mu)**2 / sigma**2
            elif prior[0].lower() == 'tophat': 
                if pval < float(prior[1][0]) or pval > float(prior[1][2]):
                    return -np.inf 
            
        return total_prior

    def lnlike(self, params):
        """
        Our likelihood. 
        """
        # TODO: Needs to be replaced with H&L approx.
        model_cls = self.model(params)
        return -0.5 * np.mat(self.bbdata - model_cls) * np.mat(self.invcov) * np.mat(self.bbdata - model_cls).T

    def lnprob(self, params):
        """
        Likelihood with priors. 
        """
        prior = self.lnpriors(params)
        if not np.isfinite(prior):
            return -np.inf
        lnprob = self.lnlike(params)
        return prior + lnprob

    def emcee_sampler(self, n_iters, nwalkers):
        """
        Sample the model with MCMC. 
        """
        # TODO: Need to save the data appropriately. 
        
        ndim = len(self.parameters.param_init)
        pos = [self.parameters.param_init * (1. + 1.e-3*np.random.randn(ndim)) for i in range(nwalkers)]

        fg_scaling, meannu = self.integrate_seds(self.parameters.param_init)
        fg_p_spectra = self.evaluate_power_spectra(self.parameters.param_init)
        np.save('scaling', fg_scaling)
        np.save('pspectra', fg_p_spectra) 
        np.save('meannu', meannu) 
 
        #sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
        #sampler.run_mcmc(pos, n_iters);
        return None


    def run(self):
        self.setup_compsep()

        if self.config['n_iters']:
            n_iters = self.config['n_iters']
        else:
            n_iters = 2**4
        if self.config['nwalkers']:
            nwalkers = self.config['nwalkers']
        else:
            nwalkers = 32

        sampler = self.emcee_sampler(n_iters, nwalkers)
            
        #np.save('big_sampling_check2', sampler.chain)
        #np.save('params', [self.parameters.param_index, self.parameters.priors])

        # this part doesn't work yet
        for out,_ in self.outputs :
            fname = self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")


if __name__ == '__main__':
    cls = PipelineStage.main()

