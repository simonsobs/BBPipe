from bbpipe import PipelineStage
from .types import FitsFile, TextFile, NumpyFile
import numpy as np
import pylab as pl
import pymaster as nmt
import healpy as hp
from fgbuster.cosmology import _get_Cl_cmb 
from fgbuster.mixingmatrix import MixingMatrix
import scipy.constants as constants
from astropy.cosmology import Planck15
from . import mk_noise_map2 as mknm
import scipy
from fgbuster.algebra import W_dB, _mmm, _mm
from fgbuster.component_model import CMB, Dust, Synchrotron
from . import V3calc as V3

def binning_definition(nside, lmin=2, lmax=200, nlb=[], custom_bins=False):
    if custom_bins:
        ells=np.arange(3*nside,dtype='int32') #Array of multipoles
        weights=(1.0/nlb)*np.ones_like(ells) #Array of weights
        bpws=-1+np.zeros_like(ells) #Array of bandpower indices
        i=0;
        while (nlb+1)*(i+1)+lmin<lmax :
            bpws[(nlb+1)*i+lmin:(nlb+1)*(i+1)+lmin]=i
            i+=1 
        ##### adding a trash bin 2<=ell<=lmin
        # bpws[lmin:(nlb+1)*i+lmin] += 1
        # bpws[2:lmin] = 0
        # weights[2:lmin]= 1.0/(lmin-2-1)
        b=nmt.NmtBin(nside,bpws=bpws, ells=ells, weights=weights)
    else:
        b=nmt.NmtBin(nside, nlb=int(1./self.config['fsky']))
    return b

def B(nu, T):
    x = constants.h * nu  *1e9/ constants.k / T
    return 2. * constants.h * (nu *1e9) ** 3 / constants.c ** 2 / np.expm1(x)

def dB(nu, T):
    x = constants.h * nu *1e9 / constants.k / T
    return B(nu, T) / T * x * np.exp(x) / np.expm1(x)

def KCMB2RJ(nu):
    return  dB(nu, Planck15.Tcmb(0).value) / (2. * (nu *1e9 / constants.c) ** 2 * constants.k)


def noise_bias_estimation(self, Cl_func, get_field_func, mask, mask_apo, 
                            w, n_cov, mask_patches, A_maxL, nhits_raw, ell_eff):
                            # ,extra_beaming=0.0):
    """
    this function performs Nsims frequency-noise simulations
    on which is applied the map-making operator estimated from
    A_maxL and the noise covariance matrix
    """
    # output operator will be of size ncomp x npixels
    if mask_patches.shape[0] == self.config['Nspec']: Npatch = mask_patches.shape[0]
    else: 
        Npatch = 1
        mask_patches = mask_patches[np.newaxis,:]
    for i_patch in range(Npatch):
        obs_pix = np.where(mask_patches[i_patch,:]!=0)[0]
        # building the (possibly pixel-dependent) mixing matrix
        A_maxL_loc = A_maxL[i_patch]

        if i_patch == 0: 
            # W is of size {n_comp, n_freqs, n_stokes, n_pixels}
            W = np.zeros((A_maxL_loc.shape[1], n_cov.shape[0], 2, n_cov.shape[-1]))
        for p in obs_pix:
            for s in range(2):
                noise_cov_inv = np.diag(1.0/n_cov[:,s,p])
                inv_AtNA = np.linalg.inv(A_maxL_loc.T.dot(noise_cov_inv).dot(A_maxL_loc))
                W[:,:,s,p] += inv_AtNA.dot(A_maxL_loc.T ).dot(noise_cov_inv)

    # can we call fgbuster.algebra.W() or fgbuster.algebra.Wd() directly?
    Cl_noise_bias = []
    for i in range(self.config['Nsims_bias']):
        # looping over simulations
        print('noise simulation # '+str(i)+' / '+str(self.config['Nsims_bias']))
        # generating frequency-maps noise simulations
        nhits, noise_maps_sim, nlev, nll = mknm.get_noise_sim(sensitivity=self.config['sensitivity_mode'], 
                        knee_mode=self.config['knee_mode'],ny_lf=self.config['ny_lf'],
                            nside_out=self.config['nside'], norm_hits_map=nhits_raw,
                                no_inh=self.config['no_inh'], CMBS4=self.config['instrument'])
                                # ,extra_beaming=extra_beaming)

        # reformating the simulated noise maps 
        noise_maps_ = np.zeros((n_cov.shape[0], 3, W.shape[-1]))
        ind = 0
        for f in range(n_cov.shape[0]): 
            for i_ in range(3): 
                noise_maps_[f,i_,:] += noise_maps_sim[ind,:]*1.0
                ind += 1
        # only keeping Q and U
        noise_maps_ = noise_maps_[:,1:,:]
        # compute Cl_noise for each frequency
        print('        -> computing noise power spectrum for each frequency map')
        if i == 0 : Cl_noise_freq = np.zeros((self.config['Nsims_bias'],noise_maps_.shape[0],len(ell_eff)))
        for f in range(noise_maps_.shape[0]):
            fn = get_field_func(mask*noise_maps_[f,0,:], mask*noise_maps_[f,1,:], mask_apo)
            Cl_noise_freq[i,f,:] = (Cl_func(fn, fn, w)[3] )
        # propagate noise through the map-making equation
        Q_noise_cmb = np.einsum('fp,fp->p', W[0,:,0,:], noise_maps_[:,0])
        U_noise_cmb = np.einsum('fp,fp->p', W[0,:,1,:], noise_maps_[:,1])
        # compute corresponding spectra
        fn = get_field_func(mask*Q_noise_cmb, mask*U_noise_cmb, mask_apo)
        Cl_noise_bias.append(Cl_func(fn, fn, w)[3] )

    return Cl_noise_bias, np.mean(Cl_noise_freq, axis=0)


def Cl_stat_res_model_func(self, freq_maps, param_beta,
                            Cl_func, get_field_func, mask, mask_apo, 
                            w, n_cov, mask_patches, Cl_noise_freq, i_cmb=0):
    '''
    This function simulate statistical foregrounds residuals
    given the noisy frequency maps and the error bar covariance, Sigma
    '''

    if mask_patches.shape[0] == self.config['Nspec']: Npatch = mask_patches.shape[0]
    else: 
        Npatch = 1
        mask_patches = mask_patches[np.newaxis,:]

    # noise_cov_inv = np.zeros_like(n_cov)
    # for p in range(Npatch):
    #     obs_pix = np.where(mask_patches[p,:]!=0)[0]
    #     for p in obs_pix:
    #         for s in range(2):
    #             noise_cov_inv[:,s,p] = 1.0/n_cov[:,s,p]

    if self.config['Nspec'] == 0: Nspec=1
    else: Nspec = self.config['Nspec']
    beta_maxL = np.zeros((Nspec,2))
    Sigma =  np.zeros((Nspec,2,2))
    for i in range(self.config['Nspec']):
        beta_maxL[i,:] = param_beta[i,:2]
        Sigma[i,:,:] = np.array([[param_beta[i,2],param_beta[i,3]],[param_beta[i,3],param_beta[i,4]]])
    # instrument = {'frequencies':np.array(self.config['frequencies'])}
    components = [CMB(), Dust(150., temp=20.0), Synchrotron(150.)]

    A = MixingMatrix(*components)
    A_ev = A.evaluator(instrument.frequency)
    A_dB_ev = A.diff_evaluator(instrument.frequency)
    comp_of_dB = A.comp_of_dB

    Cl_stat_res_model = []
    W_dB_maxL_av = []
    Sigma_av = []
    for i in range(self.config['Nsims_bias']):
        print('stat res simulation # '+str(i)+' / '+str(self.config['Nsims_bias']))
        res_map = np.zeros((6,freq_maps.shape[1]))
        for p in range(Npatch):
            # build the matrix dW/dB
            A_maxL = A_ev(beta_maxL[p])
            A_dB_maxL = A_dB_ev(beta_maxL[p])
            W_dB_maxL = W_dB(A_maxL, A_dB_maxL, comp_of_dB, invN=None)[:, i_cmb]
            # build Y which should be nbeta x npix operator
            Y = np.einsum('ij,jkl->ikl', W_dB_maxL, freq_maps*mask_patches[p,:])
            # simulate delta beta from the error covariance Sigma
            # delta_beta = np.random.multivariate_normal( np.zeros_like(Sigma[p][0,:]),
                                 # np.diag(np.diag(scipy.linalg.sqrtm(Sigma[p]))), 
                                 # size=Sigma[p].shape[0] )
            delta_beta = np.random.multivariate_normal( np.zeros_like(Sigma[p][0,:]),
                                 Sigma[p], 
                                 size=Sigma[p].shape[0] )
            if p == 0: res_map = np.diag(delta_beta).dot(Y)
            else: res_map += np.diag(delta_beta).dot(Y)
            if i ==0:
                if p == 0 :
                    W_dB_maxL_av = W_dB_maxL
                    Sigma_av = Sigma[p]
                else:
                    W_dB_maxL_av += W_dB_maxL
                    Sigma_av += Sigma[p]
        fn = get_field_func(mask*res_map[0], mask*res_map[1], mask_apo)
        Cl_stat_res_model.append(Cl_func(fn, fn, w)[3] )

    ###########    
    # debias fgs residuals from the noise
    # and finally, using the covariance of error bars on spectral indices
    # we compute the model for the statistical foregrounds residuals, 
    # cf. Errard et al 2018
    W_dB_maxL_av /= Npatch
    Sigma_av /= Npatch
    Cl_YY = np.einsum('af,fl,bf->abl', W_dB_maxL_av, Cl_noise_freq, W_dB_maxL_av)
    tr_SigmaYY = np.einsum('ij, ijl -> l', Sigma_av, Cl_YY)

    return Cl_stat_res_model, tr_SigmaYY


class BBClEstimation(PipelineStage):
    """
    Stage that performs estimate the angular power spectra of:
        * each of the sky components, foregrounds and CMB, as well as the cross terms
        * of the corresponding covariance
    """

    name='BBClEstimation'
    inputs=[('binary_mask_cut',FitsFile),('post_compsep_maps',FitsFile), ('post_compsep_cov',FitsFile),
            ('A_maxL',NumpyFile),('noise_maps',FitsFile), ('post_compsep_noise',FitsFile), 
            ('norm_hits_map', FitsFile), ('frequency_maps',FitsFile),('CMB_template_150GHz', FitsFile),
            ('mask_patches', FitsFile),('noise_cov',FitsFile), ('fitted_spectral_parameters', TextFile),
            ('Bl_eff', FitsFile), ('instrument', NumpyFile), ('W', NumpyFile)]
    
    outputs=[('Cl_clean', FitsFile),('Cl_noise', FitsFile),('Cl_cov_clean', FitsFile), 
             ('Cl_cov_freq', FitsFile), ('fsky_eff',TextFile), ('Cl_fgs', NumpyFile),
             ('Cl_CMB_template_150GHz', NumpyFile), ('mask_apo', FitsFile),
             ('Cl_noise_bias', FitsFile), ('Cl_stat_res_model', FitsFile), ('Bl_eff_', FitsFile)]

    def run(self):

        clean_map = hp.read_map(self.get_input('post_compsep_maps'),verbose=False, field=None, h=False)
        cov_map = hp.read_map(self.get_input('post_compsep_cov'),verbose=False, field=None, h=False)
        A_maxL = np.load(self.get_input('A_maxL'))
        W = np.load(self.get_input('W'))
        noise_maps=hp.read_map(self.get_input('noise_maps'),verbose=False, field=None)
        post_compsep_noise=hp.read_map(self.get_input('post_compsep_noise'),verbose=False, field=None)
        frequency_maps=hp.read_map(self.get_input('frequency_maps'),verbose=False, field=None)
        CMB_template_150GHz = hp.read_map(self.get_input('CMB_template_150GHz'), field=None)
        Bl_eff = hp.fitsfunc.read_cl(self.get_input('Bl_eff'))
        
        nhits_raw = hp.read_map(self.get_input('norm_hits_map'))
        # nhits = hp.ud_grade(nhits_raw,nside_out=self.config['nside'])
        # nh = mknm.get_mask(nhits, nside_out=self.config['nside'])
        nh=hp.ud_grade(hp.read_map(self.get_input('norm_hits_map')),nside_out=self.config['nside'])
        nhg=hp.smoothing(nh,fwhm=np.pi/180,verbose=False)
        nhg[nhg<0]=0
        # nh/=np.amax(nh)
        nhg/=np.amax(nhg)
        ZER0 = 1e-3
        # mpb=np.zeros_like(nh); mpb[nh>ZER0]=1
        mpbg=np.zeros_like(nhg); mpbg[nhg>ZER0]=1
        # print("Apodize 1")
        # msk=nmt.mask_apodization(mpb, self.config['aposize'], apotype=self.config['apotype'])
        # print("Apodize 2")
        mskg=nmt.mask_apodization(mpbg, self.config['aposize'], apotype=self.config['apotype'])


        p = np.loadtxt(self.get_input('fitted_spectral_parameters'))

        mask_patches = hp.read_map(self.get_input('mask_patches'), verbose=False, field=None)
        noise_cov=hp.read_map(self.get_input('noise_cov'),verbose=False, field=None)
        # reorganization of noise covariance 
        # instrument = {'frequencies':np.array(self.config['frequencies'])}
        # fwhm = V3.so_V3_SA_beams()
        # freqs = V3.so_V3_SA_bands()
        # instrument = {'frequencies':np.array(freqs), 'fwhm':np.array(fwhm)}
        instrument = np.load(self.get_input('instrument'), allow_pickle=True).item()

        ind = 0
        noise_cov_ = np.zeros((len(instrument.frequency), 3, frequency_maps.shape[-1]))
        for f in range(len(instrument.frequency)) : 
            for i in range(3): 
                noise_cov_[f,i,:] = noise_cov[ind,:]*1.0
                ind += 1
        # removing I from all maps
        noise_cov_ = noise_cov_[:,1:,:]


        nside_map = hp.get_nside(clean_map[0])
        print('nside_map = ', nside_map)
        
        w=nmt.NmtWorkspace()
        b = binning_definition(self.config['nside'], lmin=self.config['lmin'], lmax=self.config['lmax'],\
                                         nlb=self.config['nlb'], custom_bins=self.config['custom_bins'])

        print('building mask ... ')
        mask =  hp.read_map(self.get_input('binary_mask_cut'))
        obs_pix = np.where(mask!=0)[0]
        
        if self.config['mask_apo'] != '':
            mask_apo = hp.read_map(self.config['mask_apo'], verbose=False, field=None, h=False)
            mask_apo = hp.ud_grade(mask_apo, nside_out=self.config['nside'])
        else:
            if self.config['extra_apodization']:
                mask_apo = mskg*1.0#nmt.mask_apodization(mask, self.config['aposize'], apotype=self.config['apotype'])
            else: 
                mask_apo = mpb*1.0#mask*1.0

            if ((self.config['noise_option']!='white_noise') 
                    and (self.config['noise_option']!='no_noise')):
                # nh = hp.smoothing(nh, fwhm=1*np.pi/180.0, verbose=False) 
                # nh /= nh.max()
                # mask_apo *= nh
                mask_apo *= nhg

        fsky_eff = np.mean(mask_apo)
        print('fsky_eff = ', fsky_eff)
        np.savetxt(self.get_output('fsky_eff'), [fsky_eff])

        print('building ell_eff ... ')
        ell_eff = b.get_effective_ells()

        #Read power spectrum and provide function to generate simulated skies
        cltt,clee,clbb,clte = hp.read_cl(self.config['Cls_fiducial'])[:,:4000]
        mp_t_sim,mp_q_sim,mp_u_sim=hp.synfast([cltt,clee,clbb,clte], nside=nside_map, new=True, verbose=False)


        #This wraps up the two steps needed to compute the power spectrum
        #once the workspace has been initialized
        def compute_master(f_a,f_b,wsp) :
            cl_coupled=nmt.compute_coupled_cell(f_a,f_b)
            cl_decoupled=wsp.decouple_cell(cl_coupled)
            return cl_decoupled

        ##############################
        # simulation of the CMB
        """
        # Cl_BB_reconstructed = []
        pl.figure()
        ell_v_eff = b.get_effective_ells()
        for i in range(10):
            mp_t_sim,mp_q_sim,mp_u_sim=hp.synfast([cltt,clee,clbb,clte], nside=nside_map, new=True, verbose=False)
            # f2y0=get_field(mask*mp_q_sim,mask*mp_u_sim, purify_b=True)
            f2y0=get_field(mp_q_sim,mp_u_sim, mask_apo, purify_b=True)
            Cl_BB_reconstructed = compute_master(f2y0, f2y0, w)[3]
            pl.loglog(ell_v_eff, Cl_BB_reconstructed, 'k-', alpha=0.5)
        pl.loglog(ell_v_eff, b.bin_cell(clbb[:3*self.config['nside']]), 'r-')
        pl.savefig('test_NaMaster_simulated_CMB.pdf')
        pl.close()
        """
        # ##############################
        # ### compute the effective Bl output of component separation

        if ((self.config['common_beam_correction'] != 0.0) and (not self.config['effective_beam_correction'])):
            Bl_loc = []
            Bl_gauss_common = hp.gauss_beam( np.radians(self.config['common_beam_correction']/60), lmax=3*self.config['nside'])        
            for f in range(len(self.config['frequencies'])):
                Bl_gauss_fwhm = hp.gauss_beam( np.radians(instrument.fwhm[f]/60), lmax=3*self.config['nside'])
                Bl_loc.append( (Bl_gauss_fwhm/Bl_gauss_common)**2 )
            AtBlA = np.einsum('fi, fl, fj -> lij', np.mean(A_maxL[:,:], axis=0), np.array(Bl_loc), np.mean(A_maxL[:,:], axis=0))
            inv_AtBlA = np.linalg.inv(AtBlA)
            Bl_eff_ = np.diagonal(inv_AtBlA, axis1=-2,axis2=-1)[:,0]
            Bl_eff = np.sqrt(Bl_eff_/np.max(Bl_eff_))
            # Bl_eff *= hp.gauss_beam(np.nside2resol(self.config['nside'])/np.sqrt(8*np.log(2)*log(2)), lmax=3*self.config['nside'])
        elif self.config['effective_beam_correction']:
            # Bl_loc = []
            Bl_loc_ = []
            for f in range(len(self.config['frequencies'])):
                Bl_gauss_fwhm = hp.gauss_beam( np.radians(instrument.fwhm[f]/60), lmax=3*self.config['nside'])
                # Bl_loc.append( (1.0/Bl_gauss_fwhm)**2 )
                Bl_loc_.append( Bl_gauss_fwhm )
            # AtBlA = np.einsum('fi, fl, fj -> lij', np.mean(A_maxL, axis=0), np.array(Bl_loc), np.mean(A_maxL, axis=0))
            # AtBlA = np.einsum('fi, fl, fj -> lij', A_maxL, np.array(Bl_loc), A_maxL)
            # inv_AtBlA = np.linalg.inv(AtBlA)
            # Bl_eff_ = np.diagonal(inv_AtBlA, axis1=-2,axis2=-1)#[:,0]
            # Bl_eff = np.sqrt(Bl_eff_[:,0]/np.max(Bl_eff_))
            Bl_eff = W.dot(Bl_loc_)[0]
            # np.save('Bl_eff_', Bl_eff)
            # Bl_eff = np.sqrt(Bl_eff_/np.max(Bl_eff_))
            # Bl_eff *= hp.gauss_beam(hp.nside2resol(self.config['nside']), lmax=3*self.config['nside'])
        else:
            Bl_eff = np.zeros_like( hp.gauss_beam( 1.0/60, lmax=3*self.config['nside']) )

        ###############################
        def get_field(mp_q, mp_u, mask_apo, purify_e=False, purify_b=True) :
            #This creates a spin-2 field with both pure E and B.
            if ((self.config['common_beam_correction']!=0.0) and self.config['effective_beam_correction']):
                print('  -> common beam correction: correcting for frequency-dependent beams and convolving with a common beam')
                beam = Bl_eff
            else:beam=None
            f2y = nmt.NmtField(mask_apo, [mp_q,mp_u], purify_e=purify_e, purify_b=purify_b, beam=beam)
            return f2y

        #We initialize two workspaces for the non-pure and pure fields:
        f2y0=get_field(mp_q_sim,mp_u_sim,mask_apo)
        w.compute_coupling_matrix(f2y0,f2y0,b)
        
        ###############################
        ### compute noise bias in the comp sep maps
        Cl_cov_clean_loc = []
        Cl_cov_freq = []
        Bl_gauss_common = hp.gauss_beam( np.radians(self.config['common_beam_correction']/60), lmax=2*self.config['nside'])        
        for f in range(len(self.config['frequencies'])):
            fn = get_field(mask*noise_maps[3*f+1,:], mask*noise_maps[3*f+2,:], mask_apo)
            Cl_cov_clean_loc.append(1.0/compute_master(fn, fn, w)[3] )
            Cl_cov_freq.append(compute_master(fn, fn, w)[3])
        AtNA = np.einsum('fi, fl, fj -> lij', np.mean(A_maxL[:,:], axis=0), np.array(Cl_cov_clean_loc), np.mean(A_maxL[:,:], axis=0))
        inv_AtNA = np.linalg.inv(AtNA)
        Cl_cov_clean = np.diagonal(inv_AtNA, axis1=-2,axis2=-1)    
        Cl_cov_clean = np.vstack((ell_eff,Cl_cov_clean.swapaxes(0,1)))

        print('estimating noise bias')
        if self.config['common_beam_correction']!=0.0 : print('        with an beam appied on noise power spectra of ', self.config['common_beam_correction'], ' arcmin')
        Cl_noise_bias, Cl_noise_freq = noise_bias_estimation(self, compute_master, get_field, mask, 
                mask_apo, w, noise_cov_, mask_patches, A_maxL, nhits_raw, ell_eff)
                # , extra_beaming=self.config['common_beam_correction'])
        Cl_noise_bias = np.vstack((ell_eff, np.mean(Cl_noise_bias, axis=0), np.std(Cl_noise_bias, axis=0)))

        ### delensing if bmodes_template is provided
        if self.config['bmodes_template'] != '':
            bmodes_template = hp.read_map(self.config['bmodes_template'])
            bmodes_template = hp.ud_grade(bmodes_template, nside_out=self.config['nside'])
            clean_map[0] -= bmodes_template[0]
            clean_map[1] -= bmodes_template[1]

        ### for comparison, compute the power spectrum of the noise after comp sep
        ### compute power spectra of the cleaned sky maps
        ncomp = int(len(clean_map)/2)
        Cl_clean = [ell_eff] 
        Cl_noise = [ell_eff] 
        components = []
        print('n_comp = ', ncomp)
        ind=0
        for comp_i in range(ncomp):
            # for comp_j in range(ncomp)[comp_i:]:
            comp_j = comp_i
            print('comp_i = ', comp_i)
            print('comp_j = ', comp_j)

            components.append(str((comp_i,comp_j))) 

            ## signal spectra
            if comp_i > 1: purify_b_=False
            else: purify_b_=True

            fyp_i=get_field(mask*clean_map[2*comp_i], mask*clean_map[2*comp_i+1], mask_apo, purify_b=purify_b_)
            # fyp_i=get_field(clean_map[2*comp_i], clean_map[2*comp_i+1], purify_b=purify_b_)
            fyp_j=get_field(mask*clean_map[2*comp_j], mask*clean_map[2*comp_j+1], mask_apo, purify_b=purify_b_)
            # fyp_j=get_field(clean_map[2*comp_j], clean_map[2*comp_j+1], purify_b=purify_b_)
            Cl_clean.append(compute_master(fyp_i, fyp_j, w)[3])

            ## noise spectra
            # fyp_i_noise=get_field(mask*post_compsep_noise[2*comp_i], mask*post_compsep_noise[2*comp_i+1], purify_b=True)
            # fyp_i_noise=get_field(mask_nh*post_compsep_noise[2*comp_i], mask_nh*post_compsep_noise[2*comp_i+1], purify_b=True)
            fyp_i_noise=get_field(mask*post_compsep_noise[2*comp_i], mask*post_compsep_noise[2*comp_i+1], mask_apo, purify_b=True)
            # fyp_i_noise=get_field(post_compsep_noise[2*comp_i], post_compsep_noise[2*comp_i+1], purify_b=True)
            # fyp_j_noise=get_field(mask*post_compsep_noise[2*comp_j], mask*post_compsep_noise[2*comp_j+1], purify_b=True)
            # fyp_j_noise=get_field(mask_nh*post_compsep_noise[2*comp_j], mask_nh*post_compsep_noise[2*comp_j+1], purify_b=True)
            fyp_j_noise=get_field(mask*post_compsep_noise[2*comp_j], mask*post_compsep_noise[2*comp_j+1], mask_apo, purify_b=True)
            # fyp_j_noise=get_field(post_compsep_noise[2*comp_j], post_compsep_noise[2*comp_j+1], purify_b=True)

            Cl_noise.append(compute_master(fyp_i_noise, fyp_j_noise, w)[3])

            ind += 1

        print('ind = ', ind)
        print('shape(Cl_clean) = ', len(Cl_clean))
        print('Cl_clean = ', Cl_clean)
        print('shape(Cl_noise) = ', len(Cl_noise))
        print('all components = ', components)
        print('saving to disk ... ')
        hp.fitsfunc.write_cl(self.get_output('Cl_clean'), np.array(Cl_clean), overwrite=True)
        hp.fitsfunc.write_cl(self.get_output('Cl_noise'), np.array(Cl_noise), overwrite=True)
        hp.fitsfunc.write_cl(self.get_output('Cl_cov_clean'), np.array(Cl_cov_clean), overwrite=True)
        hp.fitsfunc.write_cl(self.get_output('Cl_cov_freq'), np.array(Cl_cov_freq), overwrite=True)
        hp.fitsfunc.write_cl(self.get_output('Cl_noise_bias'), np.array(Cl_noise_bias), overwrite=True)

        ######################
        # cross power spectra of the input frequency maps 
        # -> this is useful to estimate the statistical
        # foregrounds residuals
        ind = 0
        # instrument = {'frequencies':np.array(self.config['frequencies'])}
        frequency_maps_ = np.zeros((len(instrument.frequency), 3, frequency_maps.shape[-1]))
        for f in range(len(instrument.frequency)) : 
            for i in range(3): 
                frequency_maps_[f,i,:] =  frequency_maps[ind,:]*1.0
                ind += 1
        # removing I from all maps
        frequency_maps_ = frequency_maps_[:,1:,:]
        Nfreq = frequency_maps_.shape[0]
        Cl_fgs = np.zeros((Nfreq, Nfreq, len(ell_eff) ))
        for fi in range(Nfreq):
            for fj in range(Nfreq):
                if fi > fj:
                    Cl_fgs[fi, fj] = Cl_fgs[fj, fi]
                else:
                    fgs_i=get_field(mask*frequency_maps_[fi,0,:], mask*frequency_maps_[fi,1,:], mask_apo, purify_b=purify_b_)
                    # fgs_i=get_field(frequency_maps_[fi,0,:], frequency_maps_[fi,1,:], purify_b=purify_b_)
                    fgs_j=get_field(mask*frequency_maps_[fj,0,:], mask*frequency_maps_[fj,1,:], mask_apo, purify_b=purify_b_)
                    # fgs_j=get_field(frequency_maps_[fj,0,:], frequency_maps_[fj,1,:], purify_b=purify_b_)
                    Cl_fgs[fi,fj,:] = compute_master(fgs_i, fgs_j, w)[3]

        np.save(self.get_output('Cl_fgs'),  Cl_fgs)

        ########
        # estimation of the input CMB map cross spectrum
        cmb_i=get_field(mask*CMB_template_150GHz[1,:], mask*CMB_template_150GHz[2,:], mask_apo, purify_b=True)
        # cmb_i=get_field(CMB_template_150GHz[1,:], CMB_template_150GHz[2,:], purify_b=True)
        Cl_CMB_template_150GHz = compute_master(cmb_i, cmb_i, w)[3]
        np.save(self.get_output('Cl_CMB_template_150GHz'),  Cl_CMB_template_150GHz)

        ########
        # estimation of the modeled statistical residuals, from simulation
        if self.config['include_stat_res']:
            Cl_stat_res_model, Cl_stat_res_noise_bias = Cl_stat_res_model_func(self, frequency_maps_, p,
                            compute_master, get_field, mask, mask_apo, 
                            w, noise_cov_, mask_patches, Cl_noise_freq, i_cmb=0)
        else: 
            Cl_stat_res_model = np.zeros_like(Cl_noise_bias)
            Cl_stat_res_noise_bias = np.zeros_like(Cl_noise_bias)
        # np.save('Cl_stat_res_model', Cl_stat_res_model)
        # np.save('Cl_stat_res_noise_bias', Cl_stat_res_noise_bias)
        Cl_stat_res_model = np.vstack((ell_eff,np.mean(Cl_stat_res_model, axis=0) - Cl_stat_res_noise_bias, np.std(Cl_stat_res_model, axis=0)))
        hp.fitsfunc.write_cl(self.get_output('Cl_stat_res_model'), np.array(Cl_stat_res_model), overwrite=True)
        
        hp.fitsfunc.write_cl(self.get_output('Bl_eff_'), np.array(Bl_eff), overwrite=True)

        ########
        # outputting apodized mask
        hp.write_map(self.get_output('mask_apo'), mask_apo, overwrite=True)

if __name__ == '__main__':
    results = PipelineStage.main()
    