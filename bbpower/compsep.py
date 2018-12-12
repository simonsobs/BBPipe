from bbpipe import PipelineStage
from .types import DummyFile
from sacc.sacc import SACC
import numpy as np
import components as fgs

class BBCompSep(PipelineStage):
    """
    Template for a component separation stage
    """
    name="BBCompSep"
    inputs=[('sacc_file',SACC)]
    outputs=[('param_chains',DummyFile)]
    config_options={'foreground_model':'none'}

    def parse_sacc_file(self) :
        #Read sacc file
        self.s=SACC.loadFromHDF(self.get_input('sacc_file'))
        #This just prints information about the contents of this file
        #self.s.printInfo()

        #Get power spectrum ordering
        self.order=self.s.sortTracers()

        #Get bandpasses
        self.bpasses=[[t.z,t.Nz] for t in self.s.tracers]

        #Get bandpowers
        self.bpw_l=self.s.binning.windows[0].ls
        # We're assuming that all bandpowers are sampled at the same values of ell.
        # This is the case for the BICEP data and we may enforce it, but it is not
        # enforced within SACC.
        # we will not need this in fact. I am loading them one at a time. 
        #self.bpw_w=np.array([w.w for w in self.s.binning.windows])
        # At this point self.bpw_w is an array of shape [n_bpws,n_ells], where 
        # n_bpws is the number of power spectra stored in this file.

        #Get data vector
        self.data=self.s.mean.vector

        #Get covariance matrix
        self.covar=self.s.precision.getCovarianceMatrix()
        # TODO: At this point we haven't implemented any scale cuts, or cuts
        # on e.g. using only BB etc.
        # This could be done here with some SACC routines if needed.

        # Grabbing BB data only. 
        # Need to organize this better. 
        self.n_tracers = np.arange(12) # hard coded number of data sets.. can calculate from sacc I'm sure. 

        self.indx = []
        for t1,t2,typ,ells,ndx in order:
            if typ == b'BB':
                self.indx+=list(ndx)

        self.bbdata = self.data[indx]
        self.bbcovar = self.covar[indx][:, indx]

        # Load CMB B-mode data
        # TODO: Incorporate into the pipeline. 
        # check units 
        # strong assumption that this file is sampled at the same ells as the bandpass_l (which is true for now)
        # otherwise we will have to interpolate. That's yucky though so better to enforce this sampling. 
        cmb_bbfile = np.loadtxt('/Users/abitbol/code/self_calibration_fg/data/camb_lens_r1.dat')
        cmb_lensingfile = np.loadtxt('/Users/abitbol/code/self_calibration_fg/data/camb_lens_nobb.dat')
        
        self.cmb_ells = cmb_bbfile[:, 0]
        mask = self.cmb_ells <= self.bpw_l.max()
        self.cmb_ells = self.cmb_ells[mask] 
        self.cmb_bbr = cmb_bbfile[:, 3][mask]
        self.cmb_bblensing = cmb_lensingfile[:, 3][mask]
        return

    def model(self, params):
        r, A_s, A_d, beta_s, beta_d, alpha_s, alpha_d, epsilon = params
        
        # CMB model
        cmb_bmodes = r * self.cmb_bbr + self.cmb_bblensing
        
        # precompute SEDs
        # seds will have shape 12 = len(tns)
        fgseds = {'synch':[], 'dust': []}
        for tn in self.n_tracers:
            # integrate bandpasses 
            nus = bpasses[tn][0]
            bpass = bpasses[tn][1]

            # unit issue?
            #blackbody = fgs.blackbody(nus, TCMB)
        
            #nom_synch = fgs.normed_synch(nus, beta_s)
            #nom_dust = fgs.normed_dust(nus, beta_d)
            nom_synch = fgs.synch(nus, beta_s)
            nom_dust = fgs.dust(nus, beta_d)

            fgseds['synch'].append(np.dot(nom_synch, bpass))
            fgseds['dust'].append(np.dot(nom_dust, bpass))
        
        # precompute power laws in ell 
        # these have length 600
        nom_synch_spectrum = fgs.normed_plaw(self.bpw_l, alpha_s)
        nom_dust_spectrum = fgs.normed_plaw(self.bpw_l, alpha_d)
        nom_cross_spectrum = np.sqrt(nom_synch_spectrum * nom_dust_spectrum)
        
        cls_array_list = [] 
        for t1,t2,typ,ells,ndx in self.order:
            if typ == b'BB':
                # Integrate window functions
                # these have length 9 (number of bicep ell bins)
                windows = self.s.binning.windows[ndx]
                synch_spectrum = np.asarray([np.dot(w, nom_synch_spectrum) for w in windows])
                dust_spectrum = np.asarray([np.dot(w, nom_dust_spectrum) for w in windows])
                cross_spectrum = np.asarray([np.dot(w, nom_cross_spectrum) for w in windows])
                cmb_bb = np.asarray([np.dot(w, cmb_bmodes) for w in windows])
                
                fs1 = fgseds['synch'][t1]
                fs2 = fgseds['synch'][t2]
                fd1 = fgseds['dust'][t1]
                fd2 = fgseds['dust'][t2]
                
                synch = A_s * fs1*fs2 * synch_spectrum
                dust = A_d * fd1*fd2 * dust_spectrum
                cross = np.sqrt(A_s * A_d) * (fs1*fd2 + fs2*fd1) * cross_spectrum
                
                model = cmb_bb + synch + dust + cross
                cls_array_list.append(model)
        
        return np.asarray(cls_array_list).reshape(len(indx), ) 

    def lnpriors(self):
        # bad parameters are bad
        # break if inf
        return 
    
    def lnprob(self, params):
        model_cls = self.model(params)
        # do we need to invert this covariance matrix.....??
        return -0.5 * np.mat(self.bbdata - model_cls) * np.mat(self.bbcovar) * np.mat(self.bbdata - model_cls).T

    def lnlike(self, params):
        priors = self.lnpriors()
        lnprob = self.lnprob(params)
        return priors + lnprob

    def covfefe_sampler(self):
        # import emcee
        # emcee some shit
        # sampler.sample(stuff, model, params)

    def run(self) :
        # First, read SACC file containing reduced power spectra,
        # bandpasses, bandpowers and covariances.
        self.parse_sacc_file()
            
        #This stage currently does nothing whatsoever

        #Write outputs
        for out,_ in self.outputs :
            fname=self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")

if __name__ == '__main__':
    cls = PipelineStage.main()
