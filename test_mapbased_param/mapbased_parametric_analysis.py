from bbpipe import PipelineStage
from .types import FitsFile, TextFile, NumpyFile
import numpy as np
import matplotlib
# matplotlib.use('agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import fgbuster as fg
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.separation_recipes import weighted_comp_sep, _get_prewhiten_factors
from fgbuster.algebra import Wd
import pysm3
import copy 
import sys
from . import mk_noise_map2 as mknm
from fgbuster.observation_helpers import standardize_instrument
import healpy as hp
import sys


# def _format_alms(alms, mask_lmin=None):                                                                                                                                                                     
#     lmax = hp.Alm.getlmax(alms.shape[-1])                                                                                                                                                                   
#     alms = np.asarray(alms, order='C')                                                                                                                                                                      
#     alms = alms.view(np.float64)                                                                                                                                                                            
#     em = hp.Alm.getlm(lmax)[1]                                                                                                                                                                              
#     em = np.stack((em, em), axis=-1).reshape(-1)                                                                                                                                                            
#     mask_em = [m != 0 for m in em]                                                                                                                                                                          
#     alms[..., mask_em] *= np.sqrt(2)                                                                                                                                                                        
#     alms[..., np.arange(1, lmax+1, 2)] = hp.UNSEEN  # Mask imaginary m = 0                                                                                                                                  
#     alms = np.swapaxes(alms, 0, -1)                                                                                                                                                                         
#     if mask_lmin is not None:                                                                                                                                                                               
#         alms[mask_lmin, ...] = 0                                                                                                                                                                            
#     return alms

def _intersect_mask(maps):
    if hp.pixelfunc.is_ma(maps):
        mask = maps.mask
    else:
        mask = maps == hp.UNSEEN

    # Mask entire pixel if any of the frequencies in the pixel is masked
    return np.any(mask, axis=tuple(range(maps.ndim-1)))

def _format_alms(alms, lmin=0, nulling_option=True):                                                                                                                                                                             
    lmax = hp.Alm.getlmax(alms.shape[-1])                                                                                                                                                                   
    alms = np.asarray(alms, order='C')                                                                                                                                                                      
    alms = alms.view(np.float64)                                                                                                                                                                            
    em = hp.Alm.getlm(lmax)[1]                                                                                                                                                                              
    em = np.stack((em, em), axis=-1).reshape(-1)                                                                                                                                                            
    mask_em = [m != 0 for m in em]                                                                                                                                                           
    alms[..., mask_em] *= np.sqrt(2)                                                                                                                                                                        
    if nulling_option: 
        alms[..., np.arange(1, lmax+1, 2)] = hp.UNSEEN  # Mask imaginary m = 0                                                                                                                                  
        mask_alms = _intersect_mask(alms)                                                                                                                                                                       
        alms[..., mask_alms] = 0  # Thus no contribution to the spectral likelihood                                                                                                                             
    alms = np.swapaxes(alms, 0, -1)                                                                                                                                                                         
    if lmin != 0:                                                                                                                                                                                           
        ell = hp.Alm.getlm(lmax)[0]                                                                                                                                                                         
        ell = np.stack((ell, ell), axis=-1).reshape(-1)                                                                                                                                                     
        mask_lmin = [l < lmin for l in ell]                                                                                                                                                                 
        if nulling_option: alms[mask_lmin, ...] = hp.UNSEEN                                                                                                                                                                            
    return alms


class BBMapParamCompSep(PipelineStage):
    """
    Stage that performs three things:
        * fit of the spectral indices, using fgbuster tools, along with their error bars
        * estimate sky components, foregrounds and CMB
        * estimate components' covariance
    """
    name='BBMapParamCompSep'
    inputs= [('binary_mask_cut',FitsFile),('frequency_maps',FitsFile),('noise_cov',FitsFile),
                ('noise_maps',FitsFile),('norm_hits_map', FitsFile),('freq_maps_unbeamed',FitsFile),
                ('instrument', NumpyFile)]
    outputs=[('post_compsep_maps',FitsFile), ('post_compsep_cov',FitsFile), ('fitted_spectral_parameters',TextFile),
                 ('A_maxL',NumpyFile),('post_compsep_noise',FitsFile), ('mask_patches', FitsFile),
                 ('Bl_eff', FitsFile), ('W', NumpyFile)]

    def run(self) :
        #Read input mask
        import healpy as hp #We will want to be more general than assuming HEALPix
        binary_mask=hp.read_map(self.get_input('binary_mask_cut'),verbose=False)

        #Read frequency maps and noise covariance
        frequency_maps=hp.read_map(self.get_input('frequency_maps'),verbose=False, field=None)
        freq_maps_unbeamed=hp.read_map(self.get_input('freq_maps_unbeamed'),verbose=False, field=None)
        noise_cov=hp.read_map(self.get_input('noise_cov'),verbose=False, field=None)
        noise_maps=hp.read_map(self.get_input('noise_maps'),verbose=False, field=None)

        # reorganization of maps
        '''
        instrument = {'frequency':np.array(self.config['frequencies'])}
        if self.config['bandpass']:
            if self.config['instrument'] == 'SO':
                bandpass = 0.3*instrument['frequencies']
                print('you should check the definition of bandpasses')
                print('now is only implemented the formalism for CMB-S4')
                sys.exit()
            elif self.config['instrument'] == 'CMBS4':
                bandpass = np.array([5.0, 9.0, 12.0, 20.4, 22.8, 31.9, 34.1, 48.4, 59.4])
            # convert and renormalize bandpasses 
            inst_freq=[]
            num_steps = 100
            # [inst_freq.append((np.linspace(instrument['frequencies'][i]-bandpass[i]/2, instrument['frequencies'][i]+bandpass[i]/2, num=num_steps), \
            #             (1.0/num_steps*np.ones(num_steps)*convert_units('uK_CMB','Jysr', np.linspace(instrument['frequencies'][i]-bandpass[i]/2, instrument['frequencies'][i]+bandpass[i]/2, num=num_steps)))\
            #             /np.sum( 1.0/num_steps*np.ones(num_steps)*convert_units('uK_CMB','Jysr', np.linspace(instrument['frequencies'][i]-bandpass[i]/2, instrument['frequencies'][i]+bandpass[i]/2, num=num_steps))*(bandpass[i]/(num_steps-1)))))\
            #          for i in range(len(instrument['frequencies'])) ]
            for i in range(len(instrument['frequency'])):
                freqs_loc = np.linspace(instrument['frequency'][i]-bandpass[i]/2, instrument['frequency'][i]+bandpass[i]/2, num=num_steps)
                # bandpass_loc = (np.ones(num_steps)*\
                #     convert_units('uK_RJ','Jysr', freqs_loc))/\
                #     np.sum( np.ones(num_steps)*\
                #         convert_units('uK_RJ','Jysr', freqs_loc)*\
                #             (bandpass[i]/(num_steps-1)))
                bandpass_loc = freqs_loc**-2/(instrument['frequency'][i]*1.0)**-2
                inst_freq.append((freqs_loc, bandpass_loc))

            # [inst_freq.append((np.linspace(instrument['frequencies'][i]-bandpass[i]/2, instrument['frequencies'][i]+bandpass[i]/2, num=num_steps), \
            #             (1.0/num_steps*np.ones(num_steps)*convert_units('uK_RJ','Jysr', np.linspace(instrument['frequencies'][i]-bandpass[i]/2, instrument['frequencies'][i]+bandpass[i]/2, num=num_steps)))\
            #             /np.sum( 1.0/num_steps*np.ones(num_steps)*convert_units('uK_RJ','Jysr', np.linspace(instrument['frequencies'][i]-bandpass[i]/2, instrument['frequencies'][i]+bandpass[i]/2, num=num_steps))*(bandpass[i]/(num_steps-1)))))\
            #          for i in range(len(instrument['frequencies'])) ]
            # [inst_freq.append((np.linspace(instrument['frequencies'][i]-bandpass[i]/2, instrument['frequencies'][i]+bandpass[i]/2, num=num_steps), \
            #             (1.0/num_steps*np.ones(num_steps)))\
            #             /np.sum( 1.0/num_steps*np.ones(num_steps)*(bandpass[i]/(num_steps-1))))\
            #          for i in range(len(instrument['frequencies'])) ]
            # redefining frequencies entry to dictionary
            instr_ = copy.deepcopy(instrument)
            instr_['frequency'] = inst_freq
            instr_['channels'] = inst_freq
            instr_['channel_names'] = [str(instrument['frequency'][i]) for i in range(len(instrument['frequency']))]
            instr_['use_bandpass'] = True
            instrument_ = pysm.Instrument(instr_)
        else:
            instrument_ = copy.deepcopy(instrument)
        '''
        # instrument_ = standardize_instrument(instrument_)
        instrument_ = np.load(self.get_input('instrument'), allow_pickle=True).item()

        ind = 0
        frequency_maps_ = np.zeros((len(instrument_.frequency), 3, frequency_maps.shape[-1]))
        freq_maps_unbeamed_ = np.zeros((len(instrument_.frequency), 3, frequency_maps.shape[-1]))
        noise_maps_ = np.zeros((len(instrument_.frequency), 3, frequency_maps.shape[-1]))
        noise_cov_ = np.zeros((len(instrument_.frequency), 3, frequency_maps.shape[-1]))
        for f in range(len(instrument_.frequency)) : 
            for i in range(3): 
                frequency_maps_[f,i,:] =  frequency_maps[ind,:]*1.0
                freq_maps_unbeamed_[f,i,:] =  freq_maps_unbeamed[ind,:]*1.0
                noise_maps_[f,i,:] =  noise_maps[ind,:]*1.0
                noise_cov_[f,i,:] = noise_cov[ind,:]*1.0
                ind += 1
        

        if self.config['highpass_filtering']:
            # from scipy import signal 
            # from scipy.interpolate import interp1d
            # Nx=10000
            # window = signal.tukey(Nx, alpha=0.5)
            # window[int(Nx/2):] = window[int(Nx/2)]
            # # f = interp1d(range(Nx), window )
            # def beam_window(theta):
            #     '''
            #     providing theta in radians
            #     killing modes that are below ell ~ 60
            #     '''
            #     return f(theta*0.25*Nx/(np.pi/60))
            # np.save('freq_maps_pre_filtering', frequency_maps_)
            frequency_maps_unfiltered = frequency_maps_*1.0
            ell_knee = 100
            lmax = int(2*self.config['nside'])
            filter_window = np.array([0,]+[1.0/np.sqrt(1.0+(ell_knee*1.0/ell)**2.4) for ell in range(1,lmax)])
            print('high-pass filtering frequency maps')
            for f in range(frequency_maps_.shape[0]):
                alms = hp.map2alm(frequency_maps_[f])
                for alms_ in alms:
                    hp.almxfl(alms_, filter_window, inplace=True) 
                frequency_maps_[f] = hp.alm2map(alms, nside=self.config['nside'])

        # removing I from all maps
        frequency_maps_ = frequency_maps_[:,1:,:]
        freq_maps_unbeamed_ = freq_maps_unbeamed_[:,1:,:]
        noise_maps_ = noise_maps_[:,1:,:]
        noise_cov_ = noise_cov_[:,1:,:]

        # perform component separation
        # assuming inhomogeneous noise
        components = [CMB(), Dust(150., temp=20.0), Synchrotron(150.)]

        # options={'disp':False, 'gtol': 1e-6, 'eps': 1e-4, 'maxiter': 100, 'ftol': 1e-6 } 
        options={'disp':False, 'gtol': 1e-12, 'eps': 1e-12, 'maxiter': 100, 'ftol': 1e-12 } 
        tol=1e-18
        method='TNC'

        if self.config['Nspec']!=0.0:
            print('Nspec != 0 !! building and analysis independent Bd regions ... ')
            # read the template used to produce simulation
            # in practice, we would use the Bd map estimated from data sets.
            Bd_template = hp.read_map(self.config['path_to_dust_template'])
            # upgrade the map to the actual working resolution
            Bd_template = hp.ud_grade(Bd_template, nside_out=self.config['nside'])
            # make slices through this map. Define the regions of interest
            mask_patches = np.zeros((self.config['Nspec'], len(Bd_template)))
            # observed patches
            obs_pix = np.where(binary_mask!=0.0)[0]
            if self.config['fixed_delta_beta_slicing']:
                print(' -> fixed delta beta (from PySM) slicing in the definition of the patches')
                # thickness of the corresponding patches
                delta_Bd_patch = np.abs(np.max(Bd_template[obs_pix])-np.min(Bd_template[obs_pix]))/(self.config['Nspec'])
                # definition of slices so that it includes the max of Bd as the last step
                slices = np.arange(np.min(Bd_template[obs_pix]), np.max(Bd_template[obs_pix])+delta_Bd_patch/10.0, delta_Bd_patch )
                for i in range(self.config['Nspec']):
                    pix_within_patch = np.where((Bd_template[obs_pix] >= slices[i] ) & (Bd_template[obs_pix] < slices[i+1]))[0]
                    mask_patches[i,obs_pix[pix_within_patch]] = 1
            elif self.config['North_South_split']:
                print(' -> consider North and South patches as independent')
                # just use the observed pixels, otherwise this is time consuming
                lats = np.array([ hp.pix2ang(ipix=obs_pix[i], nside=self.config['nside'])[0] for i in range(len(obs_pix))])
                if self.config['Nspec']!= 2: 
                    print('            /!\ you cannot choose this spliting with Nspec!=2 /!\ ')
                # define South patch
                mask_patches[0,obs_pix[np.where(lats<=np.pi/2)[0]]] = 1
                # define North patch
                mask_patches[1,obs_pix[np.where(lats>=np.pi/2)[0]]] = 1
            else:
                print(' -> constant number of pixels in the definition of delta beta (from PySM) slices')
                # tune bins of a histogram so that 
                # each entry has the same number of counts
                def histedges_equalN(x, nbin):
                    npt = len(x)
                    return np.interp(np.linspace(0, npt, nbin + 1),
                                     np.arange(npt),
                                     np.sort(x))

                n, bins, _ = pl.hist(Bd_template[obs_pix], histedges_equalN(Bd_template[obs_pix], self.config['Nspec']))
                # set mask_patches 
                ind = 0
                for patch in range(len(n)):
                    pix = np.where((Bd_template[obs_pix] >= bins[ind]) & (Bd_template[obs_pix] < bins[ind+1]) )[0]
                    mask_patches[ind,obs_pix[pix]] = 1.0
                    ind += 1
            self.config['number_of_independent_patches'] = self.config['Nspec']

        elif self.config['nside_patch']!=0.0:
            print('nside_patch != 0 !! building and analysis independent Bd regions ... ')
            # make a nside_patch resolution of the binary mask
            low_res_binary_mask = hp.ud_grade(binary_mask, nside_out=self.config['nside_patch'])
            obs_low_res_pix =  np.where(low_res_binary_mask!=0.0)[0]
            self.config['number_of_independent_patches'] = len(obs_low_res_pix)
            # make a nside_patch map with the Nobs observed patches denoted from 1 to Nobs
            low_res_patch_template = np.zeros_like(low_res_binary_mask)
            low_res_patch_template[obs_low_res_pix] = np.arange(self.config['number_of_independent_patches'])
            # we udgrade this map to the actual working resolution
            patch_template = hp.ud_grade(low_res_patch_template, nside_out=self.config['nside'])
            # and apply the SO sky mask
            path_template *= binary_mask
            # make slices through this map. Define the regions of interest
            mask_patches = np.zeros((self.config['number_of_independent_patches'], len(patch_template)))
            # observed patches
            obs_pix = np.where(binary_mask!=0.0)[0]
            for i in range(self.config['number_of_independent_patches']):
                pix_within_patch = np.where(patch_template == i)[0]
                mask_patches[i,obs_pix[pix_within_patch]] = 1
        else:
            mask_patches = binary_mask[np.newaxis,:]

        noise_after_comp_sep_ = np.zeros((6, noise_cov.shape[1]))
        maps_estimated = np.zeros((6, noise_cov.shape[1]))
        cov_estimated = np.zeros((6, 6, noise_cov.shape[1]))

        # loop over pixels within defined-above regions:
        resx = []
        resS = []

        for i_patch in range(mask_patches.shape[0]):

            mask_patch_ = mask_patches[i_patch]

            # filtering masked regions of the patch ... 
            frequency_maps__ = frequency_maps_*1.0  
            frequency_maps__[:,:,np.where(mask_patch_==0)[0]] = hp.UNSEEN
            freq_maps_unbeamed__ = freq_maps_unbeamed_*1.0  
            freq_maps_unbeamed__[:,:,np.where(mask_patch_==0)[0]] = hp.UNSEEN
            noise_cov__ = noise_cov_*1.0
            noise_cov__[:,:,np.where(mask_patch_==0)[0]] = hp.UNSEEN
            
            print('actual component separation ... ')
            if self.config['harmonic_comp_sep']:
                print('     ... in harmonic space')
                lmax = 2*self.config['nside']-1

                # converting the observed frequency maps into alm (real-only arrays for fgbuster)
                for f in range(frequency_maps__.shape[0]):
                    for s in range(frequency_maps__.shape[1]):
                        alm_ = hp.map2alm(frequency_maps__[f,s], lmax=lmax)
                        alm = _format_alms(alm_, lmin=self.config['lmin'])
                        alm_noise = _format_alms(hp.map2alm(noise_maps_[f,s], lmax=lmax), lmin=self.config['lmin'])
                        if f==0 and s==0: 
                            frequency_maps__loc = np.zeros((frequency_maps__.shape[0], frequency_maps__.shape[1], len(alm)))
                            noise_maps__loc = np.zeros((frequency_maps__.shape[0], frequency_maps__.shape[1], len(alm)))
                        frequency_maps__loc[f,s] = alm
                        noise_maps__loc[f,s] = alm_noise

                # grabbing the Nl for each frequency channel
                nhits, noise_maps_sim, nlev, nll = mknm.get_noise_sim(sensitivity=self.config['sensitivity_mode'], 
                                            knee_mode=self.config['knee_mode'],ny_lf=self.config['ny_lf'],
                                                nside_out=self.config['nside'], norm_hits_map=hp.read_map(self.get_input('norm_hits_map')),
                                                    no_inh=self.config['no_inh'], CMBS4=self.config['instrument'])
                
                # building the noise covariance -- expending the Nl to every {l,m} couples
                ell_em = hp.Alm.getlm(lmax, np.arange(len(alm_)))[0]                                                                                                                                           
                ell_em = np.stack((ell_em, ell_em), axis=-1).reshape(-1) # For transformation into real alms                                                                                                            

                #Format the inverse noise matrix
                print('frequency_maps__loc.shape = ',frequency_maps__loc.shape)
                noise_cov__loc = np.ones_like(frequency_maps__loc)
                noise_cov__loc_ = np.ones((frequency_maps__loc.shape[0], frequency_maps__loc.shape[1], len(alm_)), dtype='complex')
                
                for f in range(frequency_maps__.shape[0]):
                    for s in range(frequency_maps__loc.shape[1]):
                        noise_cov__loc[f,s,:] = np.array([nll[f,l] for l in ell_em])
            else:
                print('     ... in pixel space')
                noise_cov__loc = noise_cov__*1.0
                frequency_maps__loc = frequency_maps__*1.0

            res = fg.separation_recipes.weighted_comp_sep(components, instrument_,
                         data=frequency_maps__loc, cov=noise_cov__loc, nside=self.config['nside_patch'], 
                            options=options, tol=tol, method=method)

            print('fit of spectral indices -> ', res.x)
            print('estimated error bar on spectral indices -> ', res.Sigma)

            A = MixingMatrix(*components)

            if self.config['harmonic_comp_sep']:
                print('          >>> building s = Wd in pixel space')
                A_ev = A.evaluator(instrument_.frequency)
                frequency_maps_nside = hp.get_nside(frequency_maps__[0])
                prewhiten_factors = _get_prewhiten_factors(instrument_, frequency_maps__.shape, frequency_maps_nside)
                invN = np.zeros(prewhiten_factors.shape+prewhiten_factors.shape[-1:])
                res.s = Wd(A_ev(res.x), frequency_maps__.T, invN=invN)     
                res.s = np.swapaxes(res.s,-1,0)


            resx.append(res.x)
            resS.append(res.Sigma)

            if res.s.shape[1] == 1:
                optI = 1
                optQU = 0
            elif res.s.shape[1] == 2:
                optI = 0
                optQU = 1
            else: 
                optI = 1
                optQU = 1
            
            print('re-organizing outputs of the component separation ... ')
            A_ev = A.evaluator(instrument_.frequency)
            A_maxL = A_ev(res.x)

            if i_patch == 0 :
                # defined output mixing matrix as {npatch, ncomp, nfreq}
                A_maxL_v = np.zeros((mask_patches.shape[0], A_maxL.shape[0], A_maxL.shape[1]))
            A_maxL_v[i_patch,:,:] = A_maxL*1.0

            invN = np.diag(hp.nside2resol(self.config['nside'], arcmin=True) / (instrument_.depth_p))**2
            inv_AtNA = np.linalg.inv(A_maxL.T.dot(invN).dot(A_maxL))
            W = inv_AtNA.dot( A_maxL.T ).dot(invN)
            if i_patch == 0 :
                W_v = np.zeros((mask_patches.shape[0], W.shape[0], W.shape[1]))
            W_v[i_patch,:,:] = W*1.0

            A_maxL_loc = np.zeros((2*len(instrument_.frequency), 6))

            # reshaping quantities of interest
            noise_cov_diag = np.zeros((2*len(instrument_.frequency), 2*len(instrument_.frequency), noise_cov.shape[1]))
            noise_maps__ = np.zeros((2*len(instrument_.frequency), noise_cov.shape[1]))

            for f in range(len(instrument_.frequency)):
                A_maxL_loc[2*f,:2] = A_maxL[f,0]
                A_maxL_loc[2*f+1,:2] = A_maxL[f,0]
                A_maxL_loc[2*f,2:4] = A_maxL[f,1]
                A_maxL_loc[2*f+1,2:4] = A_maxL[f,1]
                A_maxL_loc[2*f,4:] = A_maxL[f,2]
                A_maxL_loc[2*f+1,4:] = A_maxL[f,2]
                noise_cov_diag[2*f,2*f,:] = noise_cov__[f,0,:]*1.0
                noise_cov_diag[2*f+1,2*f+1,:] = noise_cov__[f,1,:]*1.0
                noise_maps__[2*f,:] = noise_maps_[f,0,:]*1.0
                noise_maps__[2*f+1,:] = noise_maps_[f,1,:]*1.0

            # define masking
            # mask_patch_ = (noise_maps__[0] == hp.UNSEEN )#| noise_maps__[0] == 0.0)
            noise_after_comp_sep = np.zeros((3,2, noise_cov.shape[1]))#*hp.UNSEEN
            obs_pix = np.where(mask_patch_==1.0)[0]

            for p in obs_pix:
                for s in range(2):
                    noise_cov_inv = np.diag(1.0/noise_cov__[:,s,p])
                    inv_AtNA = np.linalg.inv(A_maxL.T.dot(noise_cov_inv).dot(A_maxL))
                    noise_after_comp_sep[:,s,p] = inv_AtNA.dot( A_maxL.T ).dot(noise_cov_inv).dot(noise_maps_[:,s,p])
                    if self.config['common_beam_correction']!=0.0:
                        if ((p==obs_pix[0]) and (s==0)): print(' -> re-estimating res.s from unbeamed freq maps!')
                        res.s[:,s,p] = inv_AtNA.dot( A_maxL.T ).dot(noise_cov_inv).dot(freq_maps_unbeamed__[:,s,p])
                    if self.config['harmonic_comp_sep']:
                        if ((p==obs_pix[0]) and (s==0)): res.invAtNA = np.zeros((res.s.shape[0],res.s.shape[0], res.s.shape[1], res.s.shape[2]))
                        res.invAtNA[:,:,s,p] = inv_AtNA

            if self.config['highpass_filtering']:
                print('re-estimating post comp sep sky maps from un-filtered frequency maps')
                print('although using estimated spectral indices from high-pass filtered maps')
                res.s_unfiltered = np.zeros_like(res.s)
                for p in obs_pix:
                    for s in range(2):
                        noise_cov_inv = np.diag(1.0/noise_cov__[:,s,p])
                        inv_AtNA = np.linalg.inv(A_maxL.T.dot(noise_cov_inv).dot(A_maxL))
                        res.s_unfiltered[:,s,p] = inv_AtNA.dot( A_maxL.T ).dot(noise_cov_inv).dot(frequency_maps_unfiltered[:,s,p])
                res.s = res.s_unfiltered*1.0

            # the noise if the combination (sum) of the noise_after_comp_sep
            # recovered on each of the sub masks
            for f in range(noise_after_comp_sep.shape[0]):
                noise_after_comp_sep_[2*f,:] += noise_after_comp_sep[f,0,:]*1.0
                noise_after_comp_sep_[2*f+1,:] += noise_after_comp_sep[f,1,:]*1.0

            # compute the effective Bl output of component separation
            if ((self.config['common_beam_correction'] != 0.0) and (not self.config['effective_beam_correction'])):
                Bl_loc = []
                Bl_gauss_common = hp.gauss_beam( np.radians(self.config['common_beam_correction']/60), lmax=3*self.config['nside'])        
                for f in range(len(self.config['frequencies'])):
                    Bl_gauss_fwhm = hp.gauss_beam( np.radians(instrument_.fwhm[f]/60), lmax=3*self.config['nside'])
                    Bl_loc.append( Bl_gauss_common/Bl_gauss_fwhm )

                invN = np.diag(hp.nside2resol(self.config['nside'], arcmin=True) / (instrument_.depth_p))**2
                inv_AtNA = np.linalg.inv(A_maxL.T.dot(invN).dot(A_maxL))
                Bl_eff = inv_AtNA.dot( A_maxL.T ).dot(invN).dot(Bl_loc)[0]
            elif self.config['effective_beam_correction']:
                # Bl_loc2 = []
                # Bl_loc = []
                # for f in range(len(self.config['frequencies'])):
                    # print('beam = ', instrument_.fwhm[f])
                    # Bl_gauss_fwhm = hp.gauss_beam( np.radians(instrument_.fwhm[f]/60), lmax=3*self.config['nside'])
                    # Bl_loc2.append( Bl_gauss_fwhm )
                    # Bl_loc.append( Bl_gauss_fwhm )

                # print(W.shape)
                # print(np.array(Bl_loc).shape)
                # Bl_eff2_inv = np.einsum('if, fl, jf -> lij', W, np.array(Bl_loc), W)
                # Bl_eff2 = np.linalg.inv(Bl_eff2_inv)
                # Bl_eff = W.dot(np.array(Bl_loc))[0]#np.sqrt(Bl_eff2[:,0,0]/np.max(Bl_eff2[:,0,0]))
                # Bl_eff[Bl_eff!=Bl_eff] = 0.0
                # np.save('Bl_eff_v1', Bl_eff)

                # AtBlA = np.einsum('fi, fl, fj -> lij', np.mean(A_maxL, axis=0), np.array(Bl_loc2), np.mean(A_maxL, axis=0))
                # inv_AtBlA = np.linalg.inv(AtBlA)
                # Bl_eff_ = np.diagonal(inv_AtBlA, axis1=-2,axis2=-1)
                # # Bl_eff = np.sqrt(np.abs(Bl_eff_[:,0])/np.abs(np.max(Bl_eff_[:,0])))
                # Bl_eff = np.sqrt(Bl_eff_[:,0]/np.max(Bl_eff_))
                # Bl_eff[Bl_eff!=Bl_eff] = 0.0
                # np.save('Bl_eff_v2', Bl_eff)
                Bl_loc = []
                for f in range(len(self.config['frequencies'])):
                    Bl_gauss_fwhm = hp.gauss_beam( np.radians(instrument_.fwhm[f]/60), lmax=3*self.config['nside'])
                    Bl_loc.append( (1.0/Bl_gauss_fwhm)**2 )
                AtBlA = np.einsum('fi, fl, fj -> lij', np.mean(A_maxL_v, axis=0), np.array(Bl_loc), np.mean(A_maxL_v, axis=0))
                # AtBlA = np.einsum('fi, fl, fj -> lij', A_maxL, np.array(Bl_loc), A_maxL)
                inv_AtBlA = np.linalg.inv(AtBlA)
                Bl_eff_ = np.diagonal(inv_AtBlA, axis1=-2,axis2=-1)#[:,0]
                Bl_eff = np.sqrt(Bl_eff_[:,0]/np.max(Bl_eff_))
                Bl_eff[Bl_eff!=Bl_eff] = 1.0
                Bl_eff *= hp.gauss_beam( np.radians(np.mean(instrument_.fwhm)/60), lmax=3*self.config['nside'])
            else:
                Bl_eff = np.zeros_like( hp.gauss_beam( 1.0/60, lmax=3*self.config['nside']) )

            # reshape map_estimated_ from the recovered sky signals ... 
            # set to zeros areas with hp.UNSEEN
            ress = res.s[:,:,:]
            for i in range(ress.shape[0]):
                for j in range(ress.shape[1]):
                    ress[i,j,np.where(ress[i,j,:]==hp.UNSEEN)[0]] = 0.0
                    ress[i,j,np.where(ress[i,j,:]!=ress[i,j,:])[0]] = 0.0
            print('maps_estimated = ',maps_estimated)
            maps_estimated += ress.reshape((res.s.shape[0]*res.s.shape[1], res.s.shape[2]))
            for i in range(maps_estimated.shape[0]):
                print(np.where(maps_estimated[i]==hp.UNSEEN)[0])
            print('maps_estimated = ',maps_estimated)

            # reshaping and saving the covariance matrix (reorganization of the invAtNA matrix)
            # so that it is (n_stokes x n_components )^2 for each sky pixel
            cov_estimated_ = np.zeros(((res.s.shape[0]*res.s.shape[1],res.s.shape[0]*res.s.shape[1], res.s.shape[2])))
            # print('res.invAtNA.shape = ', res.invAtNA.shape)
            # print('res.s.shape = ', res.s.shape)
            # print('len(obs_pix) = ', len(obs_pix))
            # print('obs_pix = ', obs_pix)
            ind0=0
            # loop over sky components
            for c1 in range(res.invAtNA.shape[0]):
                # loop over stokes parameter
                for s1 in range(res.invAtNA.shape[2]):
                    ind1=0
                    # loop over sky components
                    for c2 in range(res.invAtNA.shape[1]):
                        # res.invAtNA[c1,c2,s1,np.where(res.invAtNA[c1,c2,s1,:]==hp.UNSEEN)[0]] = 0.0
                        # loop over stokes parameter
                        for s2 in range(res.invAtNA.shape[2]):
                            if s1==s2: cov_estimated_[ind0,ind1,obs_pix] = res.invAtNA[c1,c2,s1,obs_pix]*1.0
                            ind1+=1
                    ind0+=1
            cov_estimated += cov_estimated_
            # print('cov_estimated[0,0,:] = ', cov_estimated[0,0,:])
            print('producing map of the noise covariance')
            # pl.figure()
            cov_estimated[0,0,np.where(cov_estimated[0,0,:]==0.0)[0]]=hp.UNSEEN
            # hp.mollview(cov_estimated[0,0,:])
            # pl.savefig('noise_covariance_'+str(i_patch)+'.pdf')
            # pl.close()

        ## SAVING PRODUCTS
        np.save(self.get_output('A_maxL'), A_maxL_v)
        np.save(self.get_output('W'), W)

        hp.write_map(self.get_output('mask_patches'), mask_patches, overwrite=True)
        hp.write_map(self.get_output('post_compsep_noise'), noise_after_comp_sep_, overwrite=True)

        column_names = []
        [ column_names.extend( (('I_'+str(ch))*optI,('Q_'+str(ch))*optQU,('U_'+str(ch))*optQU)) for ch in A.components]
        column_names = [x for x in column_names if x]
        hp.write_map(self.get_output('post_compsep_maps'), maps_estimated, overwrite=True, column_names=column_names)

        hp.write_map(self.get_output('post_compsep_cov'), cov_estimated.reshape(-1, cov_estimated.shape[-1]), overwrite=True)

        column_names = []
        [column_names.append(str(A.params[i])) for i in range(len(A.params))]
        all_combinations = []
        [all_combinations.append(str(A.params[i])+' x '+str(A.params[j])) for i, j in zip(list(np.triu_indices(len(A.params))[0]),list(np.triu_indices(len(A.params))[1]) )]
        [column_names.append(all_combinations[i]) for i in range(len(all_combinations))]
        if self.config['number_of_independent_patches']!=0.0:
            column = np.hstack((resx[0],  list(resS[0][np.triu_indices(len(A.params))])))
            for p in range(self.config['number_of_independent_patches'])[1:]:
                column_ = np.hstack((resx[p],  list(resS[p][np.triu_indices(len(A.params))])))
                column = np.vstack((column, column_))

            np.savetxt(self.get_output('fitted_spectral_parameters'), column, comments=column_names)
        else:
            np.savetxt(self.get_output('fitted_spectral_parameters'), np.hstack((res.x, list(res.Sigma[np.triu_indices(len(A.params))]))), comments=column_names)

        hp.fitsfunc.write_cl(self.get_output('Bl_eff'), np.array(Bl_eff), overwrite=True)

if __name__ == '__main__':
    results = PipelineStage.main()

