import numpy as np

from bbpipe import PipelineStage
from .types import DummyFile, YamlFile
from sacc.sacc import SACC

from fgbuster import CMB, Dust, Synchrotron
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

        fg_model = self.config['fg_model']
        self.components = fg_model.keys()

        self.fg_sed_model = {}
        self.fg_nu0_norm = {}
        for component in self.components: 
            sed_name = fg_model['sed']
            sed_fnc = get_fgbuster_sed(sed_name)
            self.fg_sed_model[component] = sed_fnc(*fg_model['parameters'], units=unit)

            nu0 = fg_model[component]['parameters']['nu_0']
            self.fg_nu0_norm[component] = CMB(unit).eval(nu0) * nu0**2
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

    def model(self, params):
        """
        Defines the total model and integrates over the bandpasses and windows. 
        """
        r, A_s, A_d, beta_s, beta_d, alpha_s, alpha_d, epsilon = params

        cmb_bmodes = r * self.cmb_bbr + self.cmb_bblensing
        
        synch_seds = []
        dust_seds = []
        for tn in self.n_tracers:
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            bpass_integration = dnu*bpass

            for component in self.components: 
                #self.fg_sed_model[component]
                nom_synch = self.synch.eval(nus, beta_s) * (nus/30.)**2
                nom_dust = self.dust.eval(nus, beta_d) * (nus/353.)**2

                # units = self.fg_nu0_norm / self.cmb_norm[tn]
                # TIMES UNITS now.  not divided
                #synch_int = np.dot(nom_synch, bpass_integration) / synch_units
                #dust_int = np.dot(nom_dust, bpass_integration) / dust_units
                
                synch_seds.append(synch_int)
                dust_seds.append(dust_int)

        fgseds = {'synch':np.asarray(synch_seds), \
                  'dust':np.asarray(dust_seds)}
        
        nom_synch_spectrum = normed_plaw(self.bpw_l, alpha_s)
        nom_dust_spectrum = normed_plaw(self.bpw_l, alpha_d)
        nom_cross_spectrum = np.sqrt(nom_synch_spectrum * nom_dust_spectrum)
        
        cls_array_list = [] 
        for t1,t2,typ,ells,ndx in self.order:
            if typ == b'BB':
                windows = self.s.binning.windows[ndx]
                synch_spectrum = np.asarray([np.dot(w.w, nom_synch_spectrum) for w in windows])
                dust_spectrum = np.asarray([np.dot(w.w, nom_dust_spectrum) for w in windows])
                cross_spectrum = np.asarray([np.dot(w.w, nom_cross_spectrum) for w in windows])
                cmb_bb = np.asarray([np.dot(w.w, cmb_bmodes) for w in windows])
                
                fs1 = fgseds['synch'][t1]
                fs2 = fgseds['synch'][t2]
                fd1 = fgseds['dust'][t1]
                fd2 = fgseds['dust'][t2]
                
                synch = A_s * fs1*fs2 * synch_spectrum
                dust = A_d * fd1*fd2 * dust_spectrum
                cross = epsilon * np.sqrt(A_s * A_d) * (fs1*fd2 + fs2*fd1) * cross_spectrum
                
                model = cmb_bb + synch + dust + cross
                cls_array_list.append(model)
        
        return np.asarray(cls_array_list).reshape(len(self.indx), ) 

    def lnpriors(self, params):
        """
        Assign priors for emcee. 
        """
        # can we do anything here? 
        r, A_s, A_d, beta_s, beta_d, alpha_s, alpha_d, epsilon = params

        prior = 0
        if r < 0:
            return -np.inf
        if A_s < 0:
            return -np.inf
        if A_d < 0:
            return -np.inf
        bs0 = -3.
        prior += -0.5 * (beta_s - bs0)**2 / (0.3)**2
        bd0 = 1.6
        prior += -0.5 * (beta_d - bd0)**2 / (0.1)**2
        
        if alpha_s > 0 or alpha_s < -1.:
            return -np.inf
        if alpha_d > 0 or alpha_d < -1.:
            return -np.inf
        if np.abs(epsilon) > 1:
            return -np.inf
        return prior

    def lnlike(self, params):
        """
        Our likelihood. 
        TODO: Needs to be replaced with H&L approx.
        """
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

    def emcee_sampler(self, n_iters=2**4):
        """
        Sample the model with MCMC. 
        TODO: Need to save the data appropriately. 
        """
        ndim, nwalkers = 8, 128
        popt = [1., 1., 1., -3., 1.5, -0.5, -0.5, 0.5]
        pos = [popt * (1. + 1.e-3*np.random.randn(ndim)) for i in range(nwalkers)]
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
    #magic? 
    return sed_fnc


if __name__ == '__main__':
    cls = PipelineStage.main()

