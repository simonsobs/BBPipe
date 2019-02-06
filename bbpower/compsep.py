import numpy as np
from collections import OrderedDict

from bbpipe import PipelineStage
from .types import DummyFile, YamlFile
from sacc.sacc import SACC

from fgbuster.component_model import CMB, Dust, Synchrotron, AnalyticComponent, AME, FreeFree 
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
        # TODO: unit checks?
        self.parse_sacc_file()
        self.load_cmb()
        self.load_foregrounds()
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

            bnu=t.Nz
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
        return

    def load_foregrounds(self):
        """
        Loads the foreground models and prepares the unit conversions to K_CMB units. 
        """
        unit = 'K_RJ'
        self.get_cmb_norms(unit)

        self.fg_model = self.config['fg_model']
        self.components = self.fg_model.keys()

        self.fg_sed_model = {}
        self.fg_nu0_norm = {}
        for component in self.components: 
            sed_name = self.fg_model[component]['sed']
            sed_fnc = get_fgbuster_sed(sed_name)
            self.fg_sed_model[component] = sed_fnc(**self.fg_model[component]['parameters'], units=unit)

            nu0 = self.fg_model[component]['parameters']['nu0']
            self.fg_nu0_norm[component] = CMB(unit).eval(nu0) * nu0**2

        self.define_parameters()
        return 
            
    def get_cmb_norms(self, unit):
        """
        Evaulates the CMB unit conversion over the bandpasses. 
        """
        cmb_norms = [] 
        for tn in self.n_tracers:
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            bpass_integration = dnu*bpass

            cmb_thermo_units = CMB(unit).eval(nus) * nus**2 
            cmb_norms.append(np.dot(bpass_integration, cmb_thermo_units))
        self.cmb_norm = np.asarray(cmb_norms)
        return
    
    def define_parameters(self):
        self.param_init = []
        self.param_index = {}
        self.amp_index = {} 
        self.param_index['r'] = 0
        self.param_init.append(0)
        pindx = 1
        for component in self.components:
            for param in self.fg_model[component]['priors']:
                self.param_index[param] = pindx
                self.param_init.append(self.fg_model[component]['priors'][param][1])
                if 'amp' in param:
                    self.amp_index[component] = pindx
                pindx += 1
                # TODO: need something for cross correlation params as well 
        return

    def model(self, params):
        """
        Defines the total model and integrates over the bandpasses and windows. 
        """
        cmb_bmodes = params[0] * self.cmb_bbr + self.cmb_bblensing
        
        fg_scaling = {}
        for component in self.components:
            fg_scaling[component] = []

        for tn in self.n_tracers:
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            bpass_integration = dnu*bpass

            for component in self.components: 
                conv_rj = (nus / self.fg_model[component]['parameters']['nu0'])**2

                fg_comp_params = [] 
                for param in self.fg_sed_model[component].params:
                    pindx = self.param_index[param]
                    fg_comp_params.append(params[pindx])

                fg_sed_eval = self.fg_sed_model[component].eval(nus, *fg_comp_params) * conv_rj

                fg_units = self.fg_nu0_norm[component] / self.cmb_norm[tn]
                fg_sed_int = np.dot(fg_sed_eval, bpass_integration) * fg_units
                fg_scaling[component].append(fg_sed_int)
                
        fg_pspectra = {}
        for component in self.components:
            pspec_param_vals = []
            for param in self.fg_model[component]['spectrum']:
                pindx = self.param_index[param]
                #pspec_param_vals[param] = params[pindx]
                pspec_param_vals.append(params[pindx])
            fg_pspectra[component] = normed_plaw(self.bpw_l, *pspec_param_vals)
        
        cls_array_list = [] 
        for t1,t2,typ,ells,ndx in self.order:
            if typ == b'BB':
                windows = self.s.binning.windows[ndx]
                
                model = cmb_bmodes
                for component in self.components:
                    sed_p_scaling = fg_scaling[component][t1] * fg_scaling[component][t2]
                    model += self.amp_index[component] * sed_p_scaling * fg_pspectra[component]
                
                # TODO: Need to do something about this cross term. 
                #for cross_comp in self.config['cross'].names:
                #    cross_amp = 
                    #cross = epsilon * np.sqrt(A_s * A_d) * (fs1*fd2 + fs2*fd1) * cross_spectrum
                
                model = np.asarray([np.dot(w.w, model) for w in windows])
                cls_array_list.append(model)
        
        return np.asarray(cls_array_list).reshape(len(self.indx), ) 

    def lnpriors(self, params):
        """
        Assign priors for emcee. 
        """
        # TODO: How do we assign priors properly ?!
        
        total_prior = 0
        if params[0] < 0:
            return -np.inf
        
        for component in self.components:
            for param in self.fg_model[component]['priors']:
                prior = self.fg_model[component]['priors'][param]
                pval = params[self.param_index[param]]
                if pval < prior[0] or pval > prior[-1]:
                    return -np.inf 

        #bs0 = -3.
        #prior += -0.5 * (beta_s - bs0)**2 / (0.3)**2
        #bd0 = 1.6
        #prior += -0.5 * (beta_d - bd0)**2 / (0.1)**2
        
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

    def emcee_sampler(self, n_iters=2**4, nwalkers=128):
        """
        Sample the model with MCMC. 
        TODO: Need to save the data appropriately. 
        """
        
        ndim = len(self.param_init)
        pos = [self.param_init * (1. + 1.e-3*np.random.randn(ndim)) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
        sampler.run_mcmc(pos, n_iters);
        np.save('loadingtests', sampler.chain)
        return sampler


    def run(self) :
        self.setup_compsep()
        self.emcee_sampler(self.config['n_iter'])
            
        # this part doesn't work yet
        for out,_ in self.outputs :
            fname = self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")


def normed_plaw(ell, alpha):
    ell0 = 80.
    return (ell/ell0)**alpha 

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


if __name__ == '__main__':
    cls = PipelineStage.main()

