import numpy as np
import os
from scipy.linalg import sqrtm

from bbpipe import PipelineStage
from .types import NpzFile, SACCFile
from .fg_model import FGModel
from .param_manager import ParameterManager
from .bandpasses import Bandpass, rotate_cells, rotate_cells_mat
from fgbuster.component_model import CMB 
from sacc.sacc import SACC

from pyshtools.utils import Wigner3j
from math import pi
import scipy.special as sp #for the zeta function sp.zeta() in the 0x2 term

class BBCompSep(PipelineStage):
    """
    Component separation stage
    This stage does harmonic domain foreground cleaning (e.g. BICEP).
    The foreground model parameters are defined in the config.yml file. 
    """
    name = "BBCompSep"
    inputs = [('cells_coadded', SACCFile),('cells_noise', SACCFile),('cells_fiducial', SACCFile)]
    outputs = [('param_chains', NpzFile), ('config_copy', NpzFile)]
    config_options={'likelihood_type':'h&l', 'n_iters':32, 'nwalkers':16, 'r_init':1.e-3,
                    'sampler':'emcee', 'moments':False}

    def setup_compsep(self):
        """
        Pre-load the data, CMB BB power spectrum, and foreground models.
        """
        lmax = 384
        ells_w3j = np.arange(0, lmax)
        if self.config['moments']:
            w3j = np.zeros_like(ells_w3j, dtype=float)
            self.big_w3j = np.zeros((lmax, lmax, lmax))
            for ell1 in ells_w3j[1:]:
                for ell2 in ells_w3j[1:]:
                    w3j_array, ellmin, ellmax = Wigner3j(ell1, ell2, 0, 0, 0 )
                    w3j_array = w3j_array[:ellmax - ellmin + 1]
                    #make the w3j_array the same shape as the w3j
                    if len(w3j_array) < len(ells_w3j):
                        reference = np.zeros(len(w3j))
                        reference[:w3j_array.shape[0]] = w3j_array
                        w3j_array = reference
                    
                    w3j_array = np.concatenate([w3j[-ellmin:],w3j[:-ellmin]])
                    w3j_array = w3j_array[:len(ells_w3j)]
                    w3j_array[:ellmin] = 0

                    self.big_w3j[:,ell1,ell2] = w3j_array

            self.big_w3j = self.big_w3j**2

        self.parse_sacc_file()
        self.load_cmb()
        self.fg_model = FGModel(self.config)
        self.params = ParameterManager(self.config)
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
        #Decide if you're using H&L
        self.use_handl = self.config['likelihood_type'] == 'h&l'

        #Read data
        self.s = SACC.loadFromHDF(self.get_input('cells_coadded'))
        if self.use_handl:
            s_fid = SACC.loadFromHDF(self.get_input('cells_fiducial'), \
                                     precision_filename=self.get_input('cells_coadded'))
            s_noi = SACC.loadFromHDF(self.get_input('cells_noise'), \
                                     precision_filename=self.get_input('cells_coadded'))

        #Keep only BB measurements
        correlations=[]
        for m1 in self.config['pol_channels']:
            for m2 in self.config['pol_channels']:
                correlations.append((m1+m2).encode())

        self.s.cullType(correlations)
        self.s.cullLminLmax(self.config['l_min']*np.ones(len(self.s.tracers)),
                            self.config['l_max']*np.ones(len(self.s.tracers)))
        if self.use_handl:
            s_fid.cullType(correlations)
            s_fid.cullLminLmax(self.config['l_min']*np.ones(len(s_fid.tracers)),
                               self.config['l_max']*np.ones(len(s_fid.tracers)))
            s_noi.cullType(correlations)
            s_noi.cullLminLmax(self.config['l_min']*np.ones(len(s_noi.tracers)),
                               self.config['l_max']*np.ones(len(s_noi.tracers)))
        self.nfreqs = len(self.s.tracers)
        self.npol = len(self.config['pol_channels'])
        self.nmaps = self.nfreqs * self.npol
        self.index_ut = np.triu_indices(self.nmaps)
        self.ncross = (self.nmaps * (self.nmaps + 1)) // 2
        self.order = self.s.sortTracers()
        self.pol_order=dict(zip(self.config['pol_channels'],range(self.npol)))

        #Collect bandpasses
        self.bpss = []
        for i_t, t in enumerate(self.s.tracers):
            nu = t.z
            dnu = np.zeros_like(nu);
            dnu[1:-1] = 0.5 * (nu[2:] - nu[:-2])
            dnu[0] = nu[1] - nu[0]
            dnu[-1] = nu[-1] - nu[-2]
            bnu = t.Nz
            self.bpss.append(Bandpass(nu, dnu, bnu, i_t+1, self.config))

        #Get ell sampling
        #Avoid l<2
        mask_w = self.s.binning.windows[0].ls > 1
        self.bpw_l = self.s.binning.windows[0].ls[mask_w]
        self.n_ell = len(self.bpw_l)
        # D_ell factor
        self.dl2cl = 2 * np.pi / (self.bpw_l * (self.bpw_l + 1))
        if self.config.get('compute_dell'):
            self.dl2cl = 1.
        _,_,_,self.ell_b,_ = self.order[0]
        self.n_bpws = len(self.ell_b)
        self.windows = np.zeros([self.ncross, self.n_bpws, self.n_ell])

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
            p1,p2=typ.decode()
            ip1=self.pol_order[p1]
            ip2=self.pol_order[p2]
            # Ordering is such that polarization channel is the fastest varying index
            ind_vec=self.vector_indices[t1*self.npol + ip1, t2*self.npol + ip2]
            for b,i in enumerate(ndx):
                self.windows[ind_vec, b, :] = self.s.binning.windows[i].w[mask_w]
            v2d[:, ind_vec] = v[ndx]
            if self.use_handl:
                v2d_noi[:, ind_vec] = s_noi.mean.vector[ndx]
                v2d_fid[:, ind_vec] = s_fid.mean.vector[ndx]
            if len(ells) != self.n_bpws:
                raise ValueError("All power spectra need to be sampled at the same ells")
            for t1b, t2b, typb, ellsb, ndxb in self.order:
                p1b,p2b=typb.decode()
                ip1b=self.pol_order[p1b]
                ip2b=self.pol_order[p2b]
                ind_vecb=self.vector_indices[t1b*self.npol + ip1b, t2b*self.npol + ip2b]
                cv2d[:, ind_vec, :, ind_vecb] = cv[ndx, :][:, ndxb]

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
        cmb_lensingfile = np.loadtxt(self.config['cmb_model']['cmb_templates'][0])
        cmb_bbfile = np.loadtxt(self.config['cmb_model']['cmb_templates'][1])
        
        self.cmb_ells = cmb_bbfile[:, 0]
        mask = (self.cmb_ells <= self.bpw_l.max()) & (self.cmb_ells > 1)
        self.cmb_ells = self.cmb_ells[mask]

        # TODO: this is a patch
        nell = len(self.cmb_ells)
        self.cmb_tens = np.zeros([self.npol, self.npol, nell])
        self.cmb_lens = np.zeros([self.npol, self.npol, nell])
        self.cmb_scal = np.zeros([self.npol, self.npol, nell])
        if 'B' in self.config['pol_channels']:
            ind = self.pol_order['B']
            self.cmb_tens[ind, ind] = cmb_bbfile[:, 3][mask] - cmb_lensingfile[:, 3][mask]
            self.cmb_lens[ind, ind] = cmb_lensingfile[:, 3][mask]
        if 'E' in self.config['pol_channels']:
            ind = self.pol_order['E']
            self.cmb_tens[ind, ind] = cmb_bbfile[:, 2][mask] - cmb_lensingfile[:, 2][mask]
            self.cmb_scal[ind, ind] = cmb_lensingfile[:, 2][mask]
        return

    def integrate_seds(self, params):
        fg_scaling = np.zeros([self.fg_model.n_components, self.nfreqs])
        rot_matrices = []

        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            units = comp['cmb_n0_norm']
            sed_params = [params[comp['names_sed_dict'][k]] 
                          for k in comp['sed'].params]
            rot_matrices.append([])
            def sed(nu):
                return comp['sed'].eval(nu, *sed_params)

            for tn in range(self.nfreqs):
                sed_b, rot = self.bpss[tn].convolve_sed(sed, params)
                fg_scaling[i_c, tn] = sed_b * units
                rot_matrices[i_c].append(rot)

        return fg_scaling.T,rot_matrices

    def evaluate_power_spectra(self, params):
        fg_pspectra = np.zeros([self.fg_model.n_components,
                                self.fg_model.n_components,
                                self.npol, self.npol, self.n_ell])
        
        # Fill diagonal
        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            for cl_comb,clfunc in comp['cl'].items():
                m1, m2 = cl_comb
                ip1 = self.pol_order[m1]
                ip2 = self.pol_order[m2]
                pspec_params = [params[comp['names_cl_dict'][cl_comb][k]]
                                for k in clfunc.params]
                fg_pspectra[i_c, i_c, ip1, ip2, :] = clfunc.eval(self.bpw_l, *pspec_params) * self.dl2cl

        # Off diagonals
        for i_c1, c_name1 in enumerate(self.fg_model.component_names):
            for c_name2, epsname in self.fg_model.components[c_name1]['names_x_dict'].items():
                i_c2 = self.fg_model.component_order[c_name2]
                cl_x=np.sqrt(np.fabs(fg_pspectra[i_c1, i_c1]*
                                     fg_pspectra[i_c2, i_c2])) * params[epsname]
                fg_pspectra[i_c1, i_c2] = cl_x
                fg_pspectra[i_c2, i_c1] = cl_x

        return fg_pspectra
    
    def model(self, params):
        """
        Defines the total model and integrates over the bandpasses and windows. 
        """
        cmb_cell = (params['r_tensor'] * self.cmb_tens + \
                    params['A_lens'] * self.cmb_lens + \
                    self.cmb_scal) * self.dl2cl # [npol,npol,nell]
        fg_scaling, rot_m = self.integrate_seds(params)  # [nfreq, ncomp], [ncomp,nfreq,[matrix]]
        fg_cell = self.evaluate_power_spectra(params)  # [ncomp,ncomp,npol,npol,nell]

        # Add all components scaled in frequency (and HWP-rotated if needed)
        cls_array_fg = np.zeros([self.nfreqs,self.nfreqs,self.n_ell,self.npol,self.npol])
        fg_cell = np.transpose(fg_cell, axes = [0,1,4,2,3])  # [ncomp,ncomp,nell,npol,npol]
        cmb_cell = np.transpose(cmb_cell, axes = [2,0,1]) # [nell,npol,npol]
        for f1 in range(self.nfreqs):
            for f2 in range(f1,self.nfreqs):  # Note that we only need to fill in half of the frequencies
                cls=cmb_cell.copy()

                # Loop over component pairs
                for c1 in range(self.fg_model.n_components):
                    mat1=rot_m[c1][f1]
                    a1=fg_scaling[f1,c1]
                    for c2 in range(self.fg_model.n_components):
                        mat2=rot_m[c2][f2]
                        a2=fg_scaling[f2,c2]
                        # Rotate if needed
                        clrot=rotate_cells_mat(mat2,mat1,fg_cell[c1,c2])
                        # Scale in frequency and add
                        cls += clrot*a1*a2
                cls_array_fg[f1,f2]=cls

        # Do moments if needed        
        if self.config['moments']:
            # Evaluate 2nd order SED derivatives.            
            fg_scaling_d2 = self.integrate_seds_d2(params) # [nfreq, ncomp]
            print(fg_scaling_d2)
            print('fg_scaling_d2 shape is :', fg_scaling_d2.shape)

            # Evaluate 1st order SED derivatives.
            fg_scaling_d1 = self.integrate_seds_d1(params) # [nfreq, ncomp]
            print(fg_scaling_d1)
            print('fg_scaling_d1 shape is :', fg_scaling_d1.shape)

            # Compute 1x1 for each component
            #cls_11 = something(fg_cell, cls_betas, self.w3j) # [ncomp, nell, npol, npol]
            # Evaluate beta power spectra.
            cls_11 = np.zeros([self.fg_model.n_components, self.n_ell])
            cls_02 = np.zeros([self.fg_model.n_components, self.n_ell])
            for i_c, c_name in enumerate(self.fg_model.component_names):
                comp = self.fg_model.components[c_name]
                gamma = comp['names_moments_dict']['gamma_beta']
                amp =  comp['names_moments_dict']['amp_beta']
                cl_betas = self.bcls(lmax=384, gamma=gamma, amp=amp)
                cl_cc = fg_cell[i_c, i_c, :, 0, 0]
                cls_1x1 = self.evaluate_1x1(params, lmax=384,
                                            cls_cc=cl_cc,
                                            cls_bb=cl_betas)
                cls_11[i_c, :384] = cls_1x1
                cls_0x2 = self.evaluate_1x1(params, lmax=384,
                                            cls_cc=cl_cc,
                                            cls_bb=cl_betas)
                cls_02[i_c, :384] = cls_0x2

            exit(1)
            # Compute 0x2 for each component (essentially this is sigma_beta)
            #cls_02 = something(fg_cell, cls_betas) # [ncomp, nell, npol, npol]
            cls_02 = self.evaluate_0x2(nu, params)

            for f1 in range(self.nfreqs):
                for f2 in range(f1,self.nfreqs):  # Note that we only need to fill in half of the frequencies
                    cls = np.zeros([self.n_ell, self.npol, self.npol])
                    for c1 in range(self.fg_model.n_components):
                        cls += fg_scaling_d1[f1, c1] * fg_scaling_d1[f2, c1] * cls_11[c1]
                        cls += 0.5 * (fg_scaling_d2[f1, c1] * fg_scaling[f2, c1] +
                                      fg_scaling_d2[f2, c1] * fg_scaling[f1, c1]) * cls_02[c1]
                    cls_array_fg[f1, f2] *= cls
            
                
        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs, self.npol, self.nfreqs, self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1,self.nfreqs):
                    p0 = p1 if f1==f2 else 0
                    for p2 in range(p0,self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1,f2,:,p1,p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1!=m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        # Polarization angle rotation
        for f1 in range(self.nfreqs):
            for f2 in range(self.nfreqs):
                cls_array_list[:,f1,:,f2,:] = rotate_cells(self.bpss[f2], self.bpss[f1],
                                                           cls_array_list[:,f1,:,f2,:],
                                                           params)

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])
        
    def bcls(self, lmax, gamma, amp):
        ls = np.arange(lmax)
        bcls = np.zeros(len(ls))
        bcls[2:] = (ls[2:] / 80.)**gamma
        #crit = 2/np.log(10) #10 is the ratio between our highest and lowest frequency 300/30 GHz
        #sigma_default = crit/3 #want critical value (for convergence) =  3 sigma for the map
        #a = 4.16190627
        #b = -3.28619789
        #c = -2.56282892
        #std = a * (-gamma)**b * np.exp(c*gamma)
        #amp = (sigma_default/std)**2 
        bcls *= amp
        return bcls

    def integrate_seds_d1(self, params):
        """
        Define the first order derivative of the SED
        """
        fg_scaling_d1 = np.zeros([self.fg_model.n_components, self.nfreqs])

        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            units = comp['cmb_n0_norm']
            sed_params = [params[comp['names_sed_dict'][k]]
                for k in comp['sed'].params]

            # Set SED function with scaling beta
            def sed_d1(nu):
                #return comp['sed'].diff(nu, cls_betas, *sed_params)
                return comp['sed'].diff(nu,  *sed_params)[0] #only differentiate beta param

            for tn in range(self.nfreqs):
                sed_b = self.bpss[tn].convolve_sed(sed_d1, params)[0] #on
                fg_scaling_d1[i_c, tn] = sed_b * units
                
        return fg_scaling_d1.T


    def integrate_seds_d2(self, params):
        """
        Define the second order derivative of the SED
        """
        fg_scaling_d2 = np.zeros([self.fg_model.n_components, self.nfreqs])
        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            units = comp['cmb_n0_norm']
            sed_params = [params[comp['names_sed_dict'][k]]
                for k in comp['sed'].params]

            def sed_d2(nu):
                return comp['sed'].diff_diff(nu,  *sed_params)[0][0]

            for tn in range(self.nfreqs):
                sed_b = self.bpss[tn].convolve_sed(sed_d2, params)[0]
                fg_scaling_d2[i_c, tn] = sed_b * units
                        
        return fg_scaling_d2.T

    def evaluate_1x1(self, params, lmax, cls_cc, cls_bb):
        """
        Evaluate the 1x1 moment for auto-spectra
        """

        ls = np.arange(lmax)
        v_left = (2*ls+1) * cls_cc[:lmax]
        v_right = (2*ls+1) * cls_bb[:lmax]

        mat = self.big_w3j

        moment1x1 = np.einsum('j,ijk,k',
                              v_left,
                              mat,
                              v_right) / (4 * np.pi)
        # moment1x1 = np.dot(np.dot(mat, v_right), v_left)
        return moment1x1

    def evaluate_0x2(self, params, lmax, cls_cc, cls_bb,
                     gamma=None, amp=None):
        """
        Evaluate the 0x2 moment for auto-spectra
        Assume power law for beta
        """
        ls = np.arange(lmax)
        if gamma is not None:
            prefac = amp * (2 * zeta(-gamma-1) + zeta(-gamma) - 3) / (4 * np.pi * 80**gamma)
        else:
            np.sum( (2 * ls + 1) * cls_bb) / (4*np.pi)
        return cls_cc[:lmax] * prefac
        
    def chi_sq_dx(self, params):
        """
        Chi^2 likelihood. 
        """
        model_cls = self.model(params)
        print(self.bbdata.shape)
        print(model_cls.shape)
        return self.matrix_to_vector(self.bbdata - model_cls).flatten()

    def prepare_h_and_l(self):
        fiducial_noise = self.bbfiducial + self.bbnoise
        self.Cfl_sqrt = np.array([sqrtm(f) for f in fiducial_noise])
        self.observed_cls = self.bbdata + self.bbnoise
        return 

    def h_and_l_dx(self, params):
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
        return dx_vec

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

    def lnprob(self, par):
        """
        Likelihood with priors. 
        """
        prior = self.params.lnprior(par)
        if not np.isfinite(prior):
            return -np.inf

        params = self.params.build_params(par)
        if self.use_handl:
            dx = self.h_and_l_dx(params)
        else:
            dx = self.chi_sq_dx(params)
        like = -0.5 * np.einsum('i, ij, j',dx,self.invcov,dx)
        return prior + like

    def emcee_sampler(self):
        """
        Sample the model with MCMC. 
        """
        import emcee
        from multiprocessing import Pool
        
        fname_temp = self.get_output('param_chains')+'.h5'

        backend = emcee.backends.HDFBackend(fname_temp)

        nwalkers = self.config['nwalkers']
        n_iters = self.config['n_iters']
        ndim = len(self.params.p0)
        found_file = os.path.isfile(fname_temp)

        if not found_file:
            backend.reset(nwalkers,ndim)
            pos = [self.params.p0 + 1.e-3*np.random.randn(ndim) for i in range(nwalkers)]
            nsteps_use = n_iters
        else:
            print("Restarting from previous run")
            pos = None
            nsteps_use = max(n_iters-len(backend.get_chain()), 0)
                                    
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,backend=backend)
            if nsteps_use > 0:
                sampler.run_mcmc(pos, nsteps_use, store=True, progress=True);

        return sampler

    def minimizer(self):
        """
        Find maximum likelihood
        """
        from scipy.optimize import minimize
        def chi2(par):
            c2=-2*self.lnprob(par)
            return c2
        res=minimize(chi2, self.params.p0, method="Powell")
        return res.x

    def singlepoint(self):
        """
        Evaluate at a single point
        """
        chi2 = -2*self.lnprob(self.params.p0)
        return chi2

    def timing(self, n_eval=300):
        """
        Evaluate n times and benchmark
        """
        import time
        start = time.time()
        for i in range(n_eval):
            lik = self.lnprob(self.params.p0)
        end = time.time()

        return end-start, (end-start)/n_eval

    def run(self):
        from shutil import copyfile
        copyfile(self.get_input('config'), self.get_output('config_copy')) 
        self.setup_compsep()
        if self.config.get('sampler')=='emcee':
            sampler = self.emcee_sampler()
            np.savez(self.get_output('param_chains'),
                     chain=sampler.chain,         
                     names=self.params.p_free_names)
            print("Finished sampling")
        elif self.config.get('sampler')=='maximum_likelihood':
            sampler = self.minimizer()
            chi2 = -2*self.lnprob(sampler)
            np.savez(self.get_output('param_chains'),
                     params=sampler,
                     names=self.params.p_free_names,
                     chi2=chi2)
            print("Best fit:")
            for n,p in zip(self.params.p_free_names,sampler):
                print(n+" = %.3lE" % p)
            print("Chi2: %.3lE" % chi2)
        elif self.config.get('sampler')=='single_point':
            sampler = self.singlepoint()
            np.savez(self.get_output('param_chains'),
                     chi2=sampler,
                     names=self.params.p_free_names)
            print("Chi^2:",sampler)
        elif self.config.get('sampler')=='timing':
            sampler = self.timing()
            np.savez(self.get_output('param_chains'),
                     timing=sampler[1],
                     names=self.params.p_free_names)
            print("Total time:",sampler[0])
            print("Time per eval:",sampler[1])
        else:
            raise ValueError("Unknown sampler")

        return

if __name__ == '__main__':
    cls = PipelineStage.main()
