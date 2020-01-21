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
                            w, noise_cov, mask_patches, A_maxL, nhits):
    """
    this function performs Nsims frequency-noise simulations
    on which is applied the map-making operator estimated from
    A_maxL and the noise covariance matrix
    """
    # output operator will be of size ncomp x npixels
    W = np.zeros((A_maxL.shape[1], noise_cov.shape[-1]))
    for i_patch in range(mask_patches.shape[0]):
        obs_pix = np.where(mask_patches[i_patch,:]!=0)[0]
        # building the (possibly pixel-dependent) mixing matrix
        for p in obs_pix:
            for s in range(2):
                noise_cov_inv = np.diag(1.0/noise_cov[:,s,p])
                inv_AtNA = np.linalg.inv(A_maxL[i_patch].T.dot(noise_cov_inv).dot(A_maxL[i_patch]))
                W[:,p] = inv_AtNA.dot( A_maxL[i_patch].T ).dot(noise_cov_inv)

    # can we call fgbuster.algebra.W() or fgbuster.algebra.Wd() directly?

    Cl_noise_bias = []
    for i in range(self.conf['Nsims_bias']):
        # generating frequency-maps noise simulations
        nhits, noise_maps, nlev = mknm.get_noise_sim(sensitivity=self.config['sensitivity_mode'], 
                        knee_mode=self.config['knee_mode'],ny_lf=self.config['ny_lf'],
                            nside_out=self.config['nside'], norm_hits_map=norm_hits_map,
                                no_inh=self.config['no_inh'])
        # compute corresponding spectra
        fn = get_field_func(mask*W.dot(noise_map[::2]), mask*W.dot(noise_map[1::2]), mask_apo)
        Cl_noise_bias.append(Cl_func(fn, fn, w)[3] )

    return Cl_noise_bias



class BBClEstimation(PipelineStage):
    """
    Stage that performs estimate the angular power spectra of:
        * each of the sky components, foregrounds and CMB, as well as the cross terms
        * of the corresponding covariance
    """

    name='BBClEstimation'
    inputs=[('binary_mask_cut',FitsFile),('post_compsep_maps',FitsFile), ('post_compsep_cov',FitsFile),
            ('A_maxL',NumpyFile),('noise_maps',FitsFile), ('post_compsep_noise',FitsFile), 
            ('norm_hits_map', FitsFile), ('frequency_maps',FitsFile),('CMB_template_150GHz', FitsFile),\
            ('mask_patches', FitsFile),('noise_cov',FitsFile)]
    outputs=[('Cl_clean', FitsFile),('Cl_noise', FitsFile),('Cl_cov_clean', FitsFile), 
                ('Cl_cov_freq', FitsFile), ('fsky_eff',TextFile), ('Cl_fgs', NumpyFile),
                    ('Cl_CMB_template_150GHz', NumpyFile), ('mask_apo', FitsFile)]

    def run(self):

        clean_map = hp.read_map(self.get_input('post_compsep_maps'),verbose=False, field=None, h=False)
        cov_map = hp.read_map(self.get_input('post_compsep_cov'),verbose=False, field=None, h=False)
        A_maxL = np.load(self.get_input('A_maxL'))
        noise_maps=hp.read_map(self.get_input('noise_maps'),verbose=False, field=None)
        post_compsep_noise=hp.read_map(self.get_input('post_compsep_noise'),verbose=False, field=None)
        frequency_maps=hp.read_map(self.get_input('frequency_maps'),verbose=False, field=None)
        CMB_template_150GHz = hp.read_map(self.get_input('CMB_template_150GHz'), field=None)
        
        nhits = hp.read_map(self.get_input('norm_hits_map'))
        nhits = hp.ud_grade(nhits,nside_out=self.config['nside'])
        nh = mknm.get_mask(nhits, nside_out=self.config['nside'])

        mask_patches = hp.read_map(self.get_input('mask_patches'), verbose=False, field=None)
        noise_cov=hp.read_map(self.get_input('noise_cov'),verbose=False, field=None)
        # reorganization of noise covariance 
        instrument = {'frequencies':np.array(self.config['frequencies'])}
        ind = 0
        noise_cov_ = np.zeros((len(instrument['frequencies']), 3, frequency_maps.shape[-1]))
        for f in range(len(instrument['frequencies'])) : 
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
        mask_apo = nmt.mask_apodization(mask, self.config['aposize'], apotype=self.config['apotype'])

        if ((self.config['noise_option']!='white_noise') 
                and (self.config['noise_option']!='no_noise')):
                    # and (not self.config['no_inh'])):
            nh = hp.smoothing(nh, fwhm=1*np.pi/180.0, verbose=False) 
            nh /= nh.max()
            mask_apo *= nh

        fsky_eff = np.mean(mask_apo)
        print('fsky_eff = ', fsky_eff)
        np.savetxt(self.get_output('fsky_eff'), [fsky_eff])

        print('building ell_eff ... ')
        ell_eff = b.get_effective_ells()

        #Read power spectrum and provide function to generate simulated skies
        cltt,clee,clbb,clte = hp.read_cl(self.config['Cls_fiducial'])[:,:4000]
        mp_t_sim,mp_q_sim,mp_u_sim=hp.synfast([cltt,clee,clbb,clte], nside=nside_map, new=True, verbose=False)

        def get_field(mp_q, mp_u, mask_apo, purify_e=False, purify_b=True) :
            #This creates a spin-2 field with both pure E and B.
            f2y=nmt.NmtField(mask_apo,[mp_q,mp_u],purify_e=purify_e,purify_b=purify_b)
            return f2y

        #We initialize two workspaces for the non-pure and pure fields:
        # if ((self.config['noise_option']!='white_noise') and (self.config['noise_option']!='no_noise')):
            # f2y0=get_field(mask_nh*mp_q_sim,mask_nh*mp_u_sim)
        # else:
        f2y0=get_field(mp_q_sim,mp_u_sim,mask_apo)
        w.compute_coupling_matrix(f2y0,f2y0,b)

        #This wraps up the two steps needed to compute the power spectrum
        #once the workspace has been initialized
        def compute_master(f_a,f_b,wsp) :
            cl_coupled=nmt.compute_coupled_cell(f_a,f_b)
            cl_decoupled=wsp.decouple_cell(cl_coupled)
            return cl_decoupled

        ##############################
        # simulation of the CMB
        '''
        Cl_BB_reconstructed = []
        for i in range(10):
            mp_t_sim,mp_q_sim,mp_u_sim=hp.synfast([cltt,clee,clbb,clte], nside=nside_map, new=True, verbose=False)
            # f2y0=get_field(mask*mp_q_sim,mask*mp_u_sim, purify_b=True)
            f2y0=get_field(mp_q_sim,mp_u_sim, purify_b=True)
            Cl_BB_reconstructed.append(compute_master(f2y0, f2y0, w)[3])
        '''

        ##############################
        ### compute noise bias in the comp sep maps
        # if self.config['Nspec']==0.0:
        Cl_cov_clean_loc = []
        Cl_cov_freq = []
        for f in range(len(self.config['frequencies'])):
            fn = get_field(mask*noise_maps[3*f+1,:], mask*noise_maps[3*f+2,:], mask_apo)
            Cl_cov_clean_loc.append(1.0/compute_master(fn, fn, w)[3] )
            Cl_cov_freq.append(compute_master(fn, fn, w)[3])
        # AtNA = np.einsum('fi, fl, fj -> lij', A_maxL[0,:], np.array(Cl_cov_clean_loc), A_maxL[0,:])
        AtNA = np.einsum('fi, fl, fj -> lij', np.mean(A_maxL[:,:], axis=0), np.array(Cl_cov_clean_loc), np.mean(A_maxL[:,:], axis=0))
        inv_AtNA = np.linalg.inv(AtNA)
        Cl_cov_clean = np.diagonal(inv_AtNA, axis1=-2,axis2=-1)    
        Cl_cov_clean = np.vstack((ell_eff,Cl_cov_clean.swapaxes(0,1)))
        
        np.save('Cl_cov_clean', Cl_cov_clean)


        Cl_noise_bias = noise_bias_estimation(self, compute_master, get_field, mask, 
                mask_apo, w, noise_cov_, mask_patches, A_maxL, nhits)


        #### compute the square root of the covariance 
        # first, reshape the covariance to be square 
        # cov_map_reshaped = cov_map.reshape(int(np.sqrt(cov_map.shape[0])), int(np.sqrt(cov_map.shape[0])), cov_map.shape[-1])
        """
        # second compute the square root of it
        cov_sq = np.zeros((cov_map_reshaped.shape[0], cov_map_reshaped.shape[1], cov_map_reshaped.shape[2]))
        for p in obs_pix:
            cov_sq[:,:,p] = scipy.linalg.sqrtm(cov_map_reshaped[:,:,p])

        # perform N simulations of noise maps, with covariance cov
        Cl_cov_freq = [] 
        for i_sim in range(100):
            # generate noise following the covariance 
            noise_map_loc = np.zeros((cov_sq.shape[0],cov_sq.shape[-1]))
            for p in obs_pix:
                noise_map_loc[:,p] = cov_sq[:,:,p].dot(np.random.normal(0.0,1.0,size=cov_sq.shape[0]))
            # take power spectrum of the generated noise maps
            for c in range(int(cov_sq.shape[0]/2)):
                # Q and U for each component: e,g, CMB, dust, sync
                fn = get_field( mask*noise_map_loc[2*c,:], mask*noise_map_loc[2*c+1,:] )
                Cl_cov_freq.append(compute_master(fn, fn, w)[3])
        np.save('Cl_cov_clean_sim', Cl_cov_freq)
        Cl_cov_freq_ = Cl_cov_clean*1.0
        Cl_cov_freq_[1] = np.mean(Cl_cov_freq[::cov_sq.shape[0]], axis=0)
        Cl_cov_freq_[2] = np.mean(Cl_cov_freq[1::cov_sq.shape[0]], axis=0)
        Cl_cov_freq_[3] = np.mean(Cl_cov_freq[2::cov_sq.shape[0]], axis=0)

        np.save('Cl_cov_clean', Cl_cov_freq_)
        # pl.loglog(Cl_cov_clean[1], 'r-')
        # pl.show()
        # exit()
        """

        # simpler approach is Eq. 31 from Stompor et al 2016, 1609.03807
        # Cl_noise = 1/npix sum_pix ( AtNA_inv )
        """
        print(cov_map_reshaped[0,0,obs_pix] )
        print('------')
        print(cov_map_reshaped[1,1,obs_pix] )
        print('------')
        w_inv_Q = np.mean( cov_map_reshaped[0,0,obs_pix] )
        w_inv_U = np.mean( cov_map_reshaped[1,1,obs_pix] ) 
        # these quantities should be normalized to the pixel size
        pixel_size_in_rad = hp.nside2resol(self.config['nside'])
        print('w_inv_Q = ', w_inv_Q)
        print('w_inv_U = ', w_inv_U)
        print('Cl_cov_clean[1][0] = ', Cl_cov_clean[1][0])
        import sys
        sys.exit() 
        """

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
        print('shape(array(Cl_clean)) = ', (np.array(Cl_clean)).shape)
        print('all components = ', components)
        print('saving to disk ... ')
        hp.fitsfunc.write_cl(self.get_output('Cl_clean'), np.array(Cl_clean), overwrite=True)
        hp.fitsfunc.write_cl(self.get_output('Cl_noise'), np.array(Cl_noise), overwrite=True)
        hp.fitsfunc.write_cl(self.get_output('Cl_cov_clean'), np.array(Cl_cov_clean), overwrite=True)
        hp.fitsfunc.write_cl(self.get_output('Cl_cov_freq'), np.array(Cl_cov_freq), overwrite=True)

        ###### 
        # cross power spectra of the input frequency maps 
        # -> this is useful to estimate the statistical
        # foregrounds residuals
        ind = 0
        instrument = {'frequencies':np.array(self.config['frequencies'])}
        frequency_maps_ = np.zeros((len(instrument['frequencies']), 3, frequency_maps.shape[-1]))
        for f in range(len(instrument['frequencies'])) : 
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

        hp.write_map(self.get_output('mask_apo'), mask_apo, overwrite=True)


if __name__ == '__main__':
    results = PipelineStage.main()
    