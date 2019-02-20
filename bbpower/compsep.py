import numpy as np
from scipy.linalg import sqrtm

from bbpipe import PipelineStage
from .types import NpzFile
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
    inputs = [('cells_coadded', SACC),('cells_noise', SACC),('cells_fiducial', SACC)]
    outputs = [('param_chains', NpzFile)]
    config_options={'likelihood_type':'h&l', 'n_iters':32, 'nwalkers':16, 'r_init':1.e-3}

    def setup_compsep(self):
        """
        Pre-load the data, CMB BB power spectrum, and foreground models.
        """
        self.parse_sacc_file()
        self.load_cmb()
        self.fg_model = FGModel(self.config)
        self.parameters = FGParameters(self.config)
        if self.use_handl:
            self.prepare_h_and_l()
        return

    def matrix_to_vector(self, mat):
        return mat[..., self.index_ut[0], self.index_ut[1]]

    def vector_to_matrix(self, vec):
        if vec.ndim == 1:
            mat = np.zeros([self.nmaps, self.nmaps])
            mat[self.index_ut] = vec
            mat = mat + mat.T - np.diag(mat.diagonal())
        elif vec.ndim==2:
            mat = np.zeros([len(vec), self.nmaps, self.nmaps])
            mat[..., self.index_ut[0], self.index_ut[1]] = vec[...,:]
            for i,m in enumerate(mat):
                mat[i] = m + m.T - np.diag(m.diagonal())
        else:
            raise ValueError("Input vector can only be 1- or 2-D")
        return mat

    def parse_sacc_file(self):
        """
        Reads the data in the sacc file included the power spectra, bandpasses, and window functions. 
        """
        self.s = SACC.loadFromHDF(self.get_input('cells_coadded'))
        self.use_handl = self.config['likelihood_type'] == 'h&l'
        if self.use_handl:
            s_fid = SACC.loadFromHDF(self.get_input('cells_fiducial'), \
                                     precision_filename=self.get_input('cells_coadded'))
            s_noi = SACC.loadFromHDF(self.get_input('cells_noise'), \
                                     precision_filename=self.get_input('cells_coadded'))

        #Keep only BB measurements
        self.s.cullType(b'BB') # TODO: Modify if we want to use E
        if self.use_handl:
            s_fid.cullType(b'BB')
            s_noi.cullType(b'BB')
        self.nfreqs = len(self.s.tracers)
        self.nmaps = self.nfreqs # TODO: Modify if we want to use E
        self.index_ut = np.triu_indices(self.nmaps)
        self.ncross = (self.nmaps * (self.nmaps + 1)) // 2
        self.order = self.s.sortTracers()

        #Collect bandpasses
        self.bpasses = []
        self.meannu = []
        for t in self.s.tracers:
            nu = t.z
            dnu = np.zeros_like(nu);
            dnu[1:-1] = 0.5 * (nu[2:] - nu[:-2])
            dnu[0] = nu[1] - nu[0]
            dnu[-1] = nu[-1] - nu[-2]
            bnu = t.Nz
            self.bpasses.append([nu, dnu, bnu])
            self.meannu.append(np.sum(dnu*nu*bnu) / np.sum(dnu*bnu))

        #Get ell sampling
        self.bpw_l = self.s.binning.windows[0].ls
        _,_,_,self.ell_b,_ = self.order[0]
        self.n_bpws = len(self.ell_b)
        self.windows = np.zeros([self.ncross, self.n_bpws, len(self.bpw_l)])

        #Get power spectra and covariances
        v = self.s.mean.vector
        if len(v) != self.n_bpws * self.ncross:
            raise ValueError("C_ell vector's size is wrong")
        cv = self.s.precision.getCovarianceMatrix()

        #Parse into the right ordering
        v2d = np.zeros([self.n_bpws, self.ncross])
        if self.use_handl:
            v2d_noi = np.zeros([self.n_bpws, self.ncross])
            v2d_fid = np.zeros([self.n_bpws, self.ncross])
        cv2d = np.zeros([self.n_bpws, self.ncross, self.n_bpws, self.ncross])
        self.vector_indices = self.vector_to_matrix(np.arange(self.ncross, dtype=int)).astype(int)
        self.indx = []
        for t1,t2,typ,ells,ndx in self.order:
            for b,i in enumerate(ndx):
                self.windows[self.vector_indices[t1, t2], b, :] = self.s.binning.windows[i].w
            v2d[:,self.vector_indices[t1, t2]] = v[ndx]
            if self.use_handl:
                v2d_noi[:, self.vector_indices[t1, t2]] = s_noi.mean.vector[ndx]
                v2d_fid[:, self.vector_indices[t1, t2]] = s_fid.mean.vector[ndx]
            if len(ells) != self.n_bpws:
                raise ValueError("All power spectra need to be sampled at the same ells")
            for t1b, t2b, typb, ellsb, ndxb in self.order:
                cv2d[:, self.vector_indices[t1, t2], :, self.vector_indices[t1b, t2b]] = cv[ndx, :][:, ndxb]

        #Store data
        self.bbdata = self.vector_to_matrix(v2d)
        if self.use_handl:
            self.bbnoise = self.vector_to_matrix(v2d_noi)
            self.bbfiducial = self.vector_to_matrix(v2d_fid)
        self.bbcovar = cv2d.reshape([self.n_bpws * self.ncross, self.n_bpws * self.ncross])
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
        self.cmb_bbr -= self.cmb_bblensing
        self.get_cmb_norms()
        return

    def get_cmb_norms(self):
        """
        Evaulates the CMB unit conversion over the bandpasses. 
        """
        cmb_norms = [] 
        for tn in range(self.nfreqs):
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            bpass_integration = bpass * dnu

            cmb_thermo_units = CMB('K_RJ').eval(nus) * nus**2 
            cmb_norms.append(np.dot(bpass_integration, cmb_thermo_units))
        self.cmb_norm = np.asarray(cmb_norms)
        return
    
    def integrate_seds(self, params):
        fg_scaling = {}
        for key in self.fg_model.components:
            fg_scaling[key] = []

        for tn in range(self.nfreqs):
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            bpass_integration = bpass * dnu

            for key, component in self.fg_model.components.items(): 
                conv_rj = (nus / component['nu0'])**2

                sed_params = [] 
                for param in component['sed'].params:
                    pindx = self.parameters.param_index[param]
                    sed_params.append(params[pindx])
                
                fg_units = component['cmb_n0_norm'] / self.cmb_norm[tn]
                fg_sed_eval = component['sed'].eval(nus, *sed_params) * conv_rj
                fg_sed_int = np.dot(fg_sed_eval, bpass_integration) * fg_units
                fg_scaling[key].append(fg_sed_int)
        return fg_scaling

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
        
        cls_array_list = np.zeros([self.n_bpws,self.nmaps,self.nmaps])
        #bmodes_list = np.zeros([self.n_bpws,self.nmaps,self.nmaps])
        for t1 in range(self.nfreqs) :
            for t2 in range(t1, self.nfreqs) :
                windows = self.windows[self.vector_indices[t1, t2]]

                model = cmb_bmodes.copy()
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
                        
                model = np.dot(windows, model)
                cls_array_list[:, t1, t2] = model

                #bmodes_int = np.dot(windows, self.cmb_bblensing)        
                #bmodes_list[:, t1, t2] = bmodes_int
                if t1 != t2:
                    cls_array_list[:, t2, t1] = model
        #self.cmb_save = bmodes_list
        return cls_array_list

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

    def chi_sq_lnlike(self, params):
        """
        Chi^2 likelihood. 
        """
        model_cls = self.model(params)
        dx = self.matrix_to_vector(self.bbdata - model_cls).flatten()
        return -0.5*np.einsum('i, ij, j', dx, self.invcov, dx)

    def prepare_h_and_l(self):
        fiducial_noise = self.bbfiducial + self.bbnoise
        self.Cfl_sqrt = np.zeros_like(fiducial_noise)
        for k in range(fiducial_noise.shape[0]):
            self.Cfl_sqrt[k] = sqrtm(fiducial_noise[k])
        self.observed_cls = self.bbdata + self.bbnoise
        return 

    def h_and_l_lnlike(self, params):
        """
        Hamimeche and Lewis likelihood. 
        Taken from Cobaya written by H, L and Torrado
        See: https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/_cmblikes_prototype/_cmblikes_prototype.py
        """
        model_cls = self.model(params)
        dx_vec = []
        for k in range(model_cls.shape[0]):
            C = model_cls[k] + self.bbnoise[k]
            X = self.h_and_l(C, self.observed_cls[k], self.Cfl_sqrt[k])
            dx = self.matrix_to_vector(X).flatten()
            dx_vec = np.concatenate([dx_vec, dx])
        return -0.5 * np.einsum('i, ij, j', dx_vec, self.invcov, dx_vec)

    def h_and_l(self, C, Chat, Cfl_sqrt):
        diag, U = np.linalg.eigh(C)
        rot = U.T.dot(Chat).dot(U)
        roots = np.sqrt(diag)
        for i, root in enumerate(roots):
            rot[i, :] /= root
            rot[:, i] /= root
        U.dot(rot.dot(U.T), rot)
        diag, rot = np.linalg.eigh(rot)
        diag = np.sign(diag - 1) * np.sqrt(2 * np.maximum(0, diag - np.log(diag) - 1))
        Cfl_sqrt.dot(rot, U)
        for i, d in enumerate(diag):
            rot[:, i] = U[:, i] * d
        return rot.dot(U.T)

    def lnprob(self, params):
        """
        Likelihood with priors. 
        """
        prior = self.lnpriors(params)
        if not np.isfinite(prior):
            return -np.inf
        if self.use_handl:
            lnprob = self.h_and_l_lnlike(params)
        else:
            lnprob = self.chi_sq_lnlike(params)
        return prior + lnprob

    def emcee_sampler(self):
        """
        Sample the model with MCMC. 
        """
        zmask = self.parameters.param_init == 0
        self.parameters.param_init[zmask] += 1.e-3 * np.ones_like(zmask)
        
        nwalkers = self.config['nwalkers']
        n_iters = self.config['n_iters']
        ndim = len(self.parameters.param_init)
        pos = [self.parameters.param_init * (1. + 1.e-3*np.random.randn(ndim)) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
        sampler.run_mcmc(pos, n_iters);
        return sampler

    def make_output_dir(self):
        from datetime import datetime
        import os, errno
        from shutil import copyfile
        fmt='%Y-%m-%d-%H-%M'
        date = datetime.now().strftime(fmt)
        output_dir = './test_bbpower_minimal/outputs/'+self.config['save_prefix']+'_'+date
        try:
            os.makedirs(output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        copyfile('./test_bbpower_minimal/local_config.yml', output_dir+'/config.yml') 
        return output_dir + '/'

    def run(self):
        self.setup_compsep()
        sampler = self.emcee_sampler()

        # TODO: save things correctly
        output_dir = self.make_output_dir()
        np.save(output_dir + 'chains', sampler.chain)
        np.savez(self.get_output('param_chains'), sampler.chain)
        return

if __name__ == '__main__':
    cls = PipelineStage.main()

