from bbpipe import PipelineStage
from .types import DummyFile
from sacc.sacc import SACC
import numpy as np
import components as fgs
import emcee

class BBCompSep(PipelineStage):
    """
    Template for a component separation stage
    """
    name="BBCompSep"
    inputs=[('sacc_file',SACC)]
    outputs=[('param_chains',DummyFile)]
    # We should load all the options like this. Foreground models, parameters, priors, data sets to consider, etc etc.
    config_options={'foreground_model':{'components': ['dust', 'synch']}, 'power_spectrum_options':{'BB':True}}

    def parse_sacc_file(self) :
        self.s=SACC.loadFromHDF(self.get_input('sacc_file'))
        self.order=self.s.sortTracers()

        self.bpasses=[]
        for t in self.s.tracers :
            nu=t.z*1.e9
            #Frequency intervals
            #TODO: this is hacky. We probably want dnu to be stored by SACC too
            #      right now this is a patch caused by how the BICEP pipeline does this.
            dnu=np.zeros_like(nu);
            dnu[1:-1]=0.5*(nu[2:]-nu[:-2]);
            dnu[0]=nu[1]-nu[0]; 
            dnu[-1]=nu[-1]-nu[-2];

            bnu=t.Nz
            self.bpasses.append([nu,dnu,bnu])

        self.bpw_l=self.s.binning.windows[0].ls
        # We're assuming that all bandpowers are sampled at the same values of ell.
        # This is the case for the BICEP data and we may enforce it, but it is not
        # enforced within SACC.
        self.data=self.s.mean.vector
        self.covar=self.s.precision.getCovarianceMatrix()
        # TODO: At this point we haven't implemented any scale cuts, or cuts
        # on e.g. using only BB etc.
        # This could be done here with some SACC routines if needed.
        self.n_tracers = np.arange(12) # hard coded number of data sets.. can calculate from sacc I'm sure. 
        self.indx = []
        for t1,t2,typ,ells,ndx in self.order:
            if typ == b'BB':
                self.indx+=list(ndx)
        self.bbdata = self.data[self.indx]
        self.bbcovar = self.covar[self.indx][:, self.indx]
        self.invcov = np.linalg.solve(self.bbcovar, np.identity(len(self.bbcovar)))
        # check this inverse is good? 
        # Load CMB B-mode data
        # TODO: Incorporate loading into the pipeline?
        # check units 
        # assumption that this file is sampled at the same ells as the bandpass_l (which is true for now)
        # otherwise we will have to interpolate. That's yucky though so better to enforce this sampling. 
        cmb_bbfile = np.loadtxt('/Users/abitbol/code/self_calibration_fg/data/camb_lens_r1.dat')
        cmb_lensingfile = np.loadtxt('/Users/abitbol/code/self_calibration_fg/data/camb_lens_nobb.dat')
        
        self.cmb_ells = cmb_bbfile[:, 0]
        mask = self.cmb_ells <= self.bpw_l.max()
        self.cmb_ells = self.cmb_ells[mask] 
        self.cmb_bbr = cmb_bbfile[:, 3][mask]
        self.cmb_bblensing = cmb_lensingfile[:, 3][mask]
        return

    def precompute_units(self):
        synch_units = []
        dust_units = []
        for tn in self.n_tracers:
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            bpass_integration = dnu*bpass
    
            cmb_thermo_units = fgs.normed_cmb_thermo_units(nus)
            cmb_int = np.dot(bpass_integration, cmb_thermo_units)
            
            cmb_synch_norm = fgs.normed_cmb_thermo_units(30.e9)
            cmb_dust_norm = fgs.normed_cmb_thermo_units(353.e9)
            
            synch_units.append(cmb_int / cmb_synch_norm)
            dust_units.append(cmb_int / cmb_dust_norm)

        self.cmb_units = {'synch_units':np.asarray(synch_units), \
                          'dust_units':np.asarray(dust_units)}
        return
        

    def model(self, params):
        # TODO: generalize 
        # break into SED and harmonic terms

        r, A_s, A_d, beta_s, beta_d, alpha_s, alpha_d, epsilon = params
        
        # CMB model
        cmb_bmodes = r * self.cmb_bbr + self.cmb_bblensing
        
        # precompute SEDs
        # seds will have shape 12 = len(tns)
        synch_seds = []
        dust_seds = []
        for tn in self.n_tracers:
            # integrate bandpasses 
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            bpass_integration = dnu*bpass

            nom_synch = fgs.normed_synch(nus, beta_s)
            nom_dust = fgs.normed_dust(nus, beta_d)

            synch_units = self.cmb_units['synch_units'][tn]
            dust_units = self.cmb_units['dust_units'][tn]

            synch_int = np.dot(nom_synch, bpass_integration) / synch_units
            dust_int = np.dot(nom_dust, bpass_integration) / dust_units
            
            synch_seds.append(synch_int)
            dust_seds.append(dust_int)

        fgseds = {'synch':np.asarray(synch_seds), \
                  'dust':np.asarray(dust_seds)}
        
        # precompute power laws in ell 
        nom_synch_spectrum = fgs.normed_plaw(self.bpw_l, alpha_s)
        nom_dust_spectrum = fgs.normed_plaw(self.bpw_l, alpha_d)
        nom_cross_spectrum = np.sqrt(nom_synch_spectrum * nom_dust_spectrum)
        
        cls_array_list = [] 
        for t1,t2,typ,ells,ndx in self.order:
            if typ == b'BB':
                # Integrate window functions
                # these have length 9 (number of bicep ell bins)
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
        # bad parameters are bad
        r, A_s, A_d, beta_s, beta_d, alpha_s, alpha_d, epsilon = params
        
        if A_s < 0:
            return -np.inf
        if A_d < 0:
            return -np.inf
        if r < 0:
            return -np.inf
        if np.abs(epsilon) > 1:
            return -np.inf
        return 0.
    
    def lnlike(self, params):
        model_cls = self.model(params)
        return -0.5 * np.mat(self.bbdata - model_cls) * np.mat(self.invcov) * np.mat(self.bbdata - model_cls).T

    def lnprob(self, params):
        prior = self.lnpriors(params)
        if not np.isfinite(prior):
            return -np.inf
        lnprob = self.lnlike(params)
        return prior + lnprob

    def emcee_LOL_sampler(self):
        ndim, nwalkers = 8, 128
        popt = [1., 1., 1., -3., 1.5, -0.5, -0.5, 0.5]
        pos = [popt * (1. + 1.e-3*np.random.randn(ndim)) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=[params])
        sampler.run_mcmc(pos, 2**5);
        return sampler


    def run(self) :
        # First, read SACC file containing reduced power spectra,
        # bandpasses, bandpowers and covariances.
        self.parse_sacc_file()
        self.precompute_units()
            
        #Write outputs
        for out,_ in self.outputs :
            fname=self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")

if __name__ == '__main__':
    cls = PipelineStage.main()
