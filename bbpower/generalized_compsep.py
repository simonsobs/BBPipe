from bbpipe import PipelineStage
from .types import DummyFile
from sacc.sacc import SACC
import numpy as np
import components as fgs
from fgbuster import CMB, Dust, Synchrotron
import emcee

class BBCompSep(PipelineStage):
    """
    Component separation stage
    """
    name = "BBCompSep"
    inputs = [('sacc_file', SACC)]
    outputs = [('param_chains', DummyFile)]
    config_options = {'foreground_model':{'components': ['dust', 'synch']}, 'power_spectrum_options':{'BB':True}}

    def setup_compsep(self):
        self.parse_sacc_file()
        self.load_cmb()
        self.load_foregrounds()
        return

    def parse_sacc_file(self):
        # We're assuming that all bandpowers are sampled at the same values of ell.
        self.s = SACC.loadFromHDF(self.get_input('sacc_file'))
        self.order = self.s.sortTracers()

        self.bpasses = []
        for t in self.s.tracers :
            nu = t.z
            #nu = t.z*1.e9
            dnu = np.zeros_like(nu);
            dnu[1:-1] = 0.5 * (nu[2:] - nu[:-2])
            dnu[0] = nu[1] - nu[0]
            dnu[-1] = nu[-1] - nu[-2]

            bnu=t.Nz
            self.bpasses.append([nu, dnu, bnu])

        self.bpw_l = self.s.binning.windows[0].ls
        self.data = self.s.mean.vector
        #self.covar = self.s.precision.getCovarianceMatrix()
        self.covar = self.s.precision.getCovarianceMatrix().reshape((2700, 2700))
        self.n_tracers = np.arange(12) 
        self.indx = []
        for t1,t2,typ,ells,ndx in self.order:
            if typ == b'BB':
                self.indx + =list(ndx)
        self.bbdata = self.data[self.indx]
        self.bbcovar = self.covar[self.indx][:, self.indx]
        self.invcov = np.linalg.solve(self.bbcovar, np.identity(len(self.bbcovar)))
        return

    def load_cmb(self):
        #load these file names from inputs
        cmb_bbfile = np.loadtxt('/Users/abitbol/code/self_calibration_fg/data/camb_lens_r1.dat')
        cmb_lensingfile = np.loadtxt('/Users/abitbol/code/self_calibration_fg/data/camb_lens_nobb.dat')
        
        self.cmb_ells = cmb_bbfile[:, 0]
        mask = self.cmb_ells <= self.bpw_l.max()
        self.cmb_ells = self.cmb_ells[mask] 
        self.cmb_bbr = cmb_bbfile[:, 3][mask]
        self.cmb_bblensing = cmb_lensingfile[:, 3][mask]
        return

    def load_foregrounds(self):
        # load here 
        synch_units = []
        dust_units = []
        for tn in self.n_tracers:
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            bpass_integration = dnu*bpass
    
            cmb_thermo_units = CMB('K_RJ').eval(nus) * nus**2 
            cmb_int = np.dot(bpass_integration, cmb_thermo_units)
            
            # fuck me, needs fixing, probably load from config_options
            cmb_synch_norm = CMB('K_RJ').eval(30.) * (30.**2)
            cmb_dust_norm = CMB('K_RJ').eval(353.) * (353.**2)
            
            synch_units.append(cmb_int / cmb_synch_norm)
            dust_units.append(cmb_int / cmb_dust_norm)

        self.cmb_units = {'synch_units':np.asarray(synch_units), \
                          'dust_units':np.asarray(dust_units)}

        self.synch = Synchrotron(30., units='K_RJ')
        self.dust = Dust(353., temp=19.6, units='K_RJ')
        return
        

    def model(self, params):
        # load here? 
        r, A_s, A_d, beta_s, beta_d, alpha_s, alpha_d, epsilon = params
        
        cmb_bmodes = r * self.cmb_bbr + self.cmb_bblensing
        
        synch_seds = []
        dust_seds = []
        for tn in self.n_tracers:
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            bpass_integration = dnu*bpass

            nom_synch = self.synch.eval(nus, beta_s) * (nus/30.)**2
            nom_dust = self.dust.eval(nus, beta_d) * (nus/353.)**2

            synch_units = self.cmb_units['synch_units'][tn]
            dust_units = self.cmb_units['dust_units'][tn]

            synch_int = np.dot(nom_synch, bpass_integration) / synch_units
            dust_int = np.dot(nom_dust, bpass_integration) / dust_units
            
            synch_seds.append(synch_int)
            dust_seds.append(dust_int)

        fgseds = {'synch':np.asarray(synch_seds), \
                  'dust':np.asarray(dust_seds)}
        
        nom_synch_spectrum = fgs.normed_plaw(self.bpw_l, alpha_s)
        nom_dust_spectrum = fgs.normed_plaw(self.bpw_l, alpha_d)
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
        model_cls = self.model(params)
        return -0.5 * np.mat(self.bbdata - model_cls) * np.mat(self.invcov) * np.mat(self.bbdata - model_cls).T

    def lnprob(self, params):
        prior = self.lnpriors(params)
        if not np.isfinite(prior):
            return -np.inf
        lnprob = self.lnlike(params)
        return prior + lnprob

    def emcee_sampler(self, n_iters=2**4):
        ndim, nwalkers = 8, 128
        popt = [1., 1., 1., -3., 1.5, -0.5, -0.5, 0.5]
        pos = [popt * (1. + 1.e-3*np.random.randn(ndim)) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=[params])
        sampler.run_mcmc(pos, n_iters);
        np.save('somecoolfilename_samplerchain', sampler.chain)
        return sampler


    def run(self) :
        self.setup_compsep()
        self.emcee_sampler(n_iters)
            
        # this part doesn't work yet
        for out,_ in self.outputs :
            fname = self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")

if __name__ == '__main__':
    cls = PipelineStage.main()

