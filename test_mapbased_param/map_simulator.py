from bbpipe import PipelineStage
from .types import FitsFile, TextFile, NumpyFile
import numpy as np
import matplotlib

import matplotlib.pyplot as pl
import pysm3 as pysm
from pysm3 import models
from . import mk_noise_map2 as mknm
from . import V3calc as V3
import healpy as hp
import copy
import glob
import os
import scipy

import fgbuster
from fgbuster.observation_helpers import get_instrument, get_sky, get_observation, standardize_instrument


import sys
sys.path.append('/global/cfs/cdirs/sobs/users/krach/BBSims/NOISE_20201207/')
from combine_noise import *


def noise_covariance_estimation(self, map_shape, instrument, nhits):
    """
    Estimation of the noise covariance matrix
    """
    noise_cov = np.zeros(map_shape)
    noise_cov_beamed = np.zeros(map_shape)
    nhits_nz = np.where(nhits!=0)[0]

    for i_sim in range(self.config['Nsims_bias']):

        if self.config['external_noise_sims']!='' or self.config['Nico_noise_combination']:
            noise_maps = np.zeros(map_shape)
            # print('noise_maps.shape = ', noise_maps.shape)
            print('NOISE COV ESTIMATION LOADING EXTERNAL NOISE-ONLY MAPS, SIM#',i_sim,'/',self.config['Nsims_bias'])

            if self.config['Nico_noise_combination']:
                if self.config['knee_mode'] == 2 : knee_mode_loc = None
                else: knee_mode_loc = self.config['knee_mode']
                factors = compute_noise_factors(self.config['sensitivity_mode'], knee_mode_loc)

            for f in range(len(instrument.frequency)):
                print('loading noise map for frequency ', str(int(instrument.frequency[f])))
                # noise_maps[3*f:3*(f+1),:] = hp.ud_grade(hp.read_map(list_of_files[f], field=None), nside_out=self.config['nside'])

                if self.config['Nico_noise_combination']:
                    noise_loc = combine_noise_maps(i_sim, instrument.frequency[f], factors)
                else:
                    noise_loc = hp.read_map(glob.glob(os.path.join(self.config['external_noise_sims'],'SO_SAT_'+str(int(instrument.frequency[f]))+'_noise_FULL_*_white_20201207.fits'))[0], field=None)

                alms = hp.map2alm(noise_loc, lmax=3*self.config['nside'])
                Bl_gauss_pix = hp.gauss_beam( hp.nside2resol(self.config['nside']), lmax=2*self.config['nside'])        
                for alm_ in alms: hp.almxfl(alm_, Bl_gauss_pix, inplace=True)             
                noise_maps[3*f:3*(f+1),:] = hp.alm2map(alms, self.config['nside'])  

                if ((not self.config['no_inh']) and (self.config['Nico_noise_combination'])):
                    # renormalize the noise map to take into account the effect of inhomogeneous noise
                    noise_maps[3*f:3*(f+1),nhits_nz] /= np.sqrt(nhits[nhits_nz]/np.max(nhits[nhits_nz]))

        elif self.config['noise_option']=='white_noise':
            np.random.seed(i_sim)
            nlev_map = np.zeros(map_shape)
            for f in range(len(instrument.frequency)):
                nlev_map[3*f:3*f+3,:] = np.array([instrument.depth_i[f], instrument.depth_p[f], instrument.depth_p[f]])[:,np.newaxis]*np.ones((3,map_shape[-1]))
            nlev_map /= hp.nside2resol(self.config['nside'], arcmin=True)
            noise_maps = np.random.normal(freq_maps*0.0, nlev_map, map_shape)

        elif self.config['noise_option']=='no_noise': 
            pass

        noise_maps_beamed = noise_maps*1.0

        if self.config['common_beam_correction']!=0.0:

            Bl_gauss_common = hp.gauss_beam( np.radians(self.config['common_beam_correction']/60), lmax=2*self.config['nside'])        
            for f in range(len(instrument.frequency)):
                Bl_gauss_fwhm = hp.gauss_beam( np.radians(instrument.fwhm[f]/60), lmax=2*self.config['nside'])

                alms_n = hp.map2alm(noise_maps_beamed[3*f:3*(f+1),:], lmax=3*self.config['nside'])
                for alms_ in alms_n:
                    hp.almxfl(alms_, Bl_gauss_common/Bl_gauss_fwhm, inplace=True)             
                noise_maps_beamed[3*f:3*(f+1),:] = hp.alm2map(alms_n, self.config['nside'])   

        noise_cov += noise_maps**2
        noise_cov_beamed += noise_maps_beamed**2

    return noise_cov/self.config['Nsims_bias'], noise_cov_beamed/self.config['Nsims_bias']


def pp_noise_covariance_estimation(self, binary_mask):
    """
    Estimation of the pixel-based covariance matrix
    """

    for i in range(self.config['Nsims_bias']):
        # looping over simulations
        print('noise simulation # '+str(i)+' / '+str(self.config['Nsims_bias']))
        # generating frequency-maps noise simulations
        nhits, noise_maps_sim, nlev = mknm.get_noise_sim(sensitivity=self.config['sensitivity_mode'], 
                        knee_mode=self.config['knee_mode'],ny_lf=self.config['ny_lf'],
                            nside_out=self.config['nside'], norm_hits_map=hp.read_map(self.get_input('norm_hits_map')),
                                no_inh=self.config['no_inh'], CMBS4=self.config['instrument'])

        noise_maps_sim[:,np.where(binary_mask==0)[0]]=0.0
        good_pix = np.where(binary_mask==1)[0]
        Nfreqs = int(noise_maps_sim.shape[0]/3)

        if i == 0: Ncov = np.zeros((Nfreqs, 2*len(good_pix), 2*len(good_pix)))
        for f in range(Nfreqs):
            for u in range(2):
                for q in range(2):
                    if u==0 and q==0: Ncov[f,::2,::2] += np.outer( noise_maps_sim[2*f+u+1,good_pix], noise_maps_sim[2*f+q+1,good_pix] )
                    elif u==0 and q==1: Ncov[f,::2,1::2] += np.outer( noise_maps_sim[2*f+u+1,good_pix], noise_maps_sim[2*f+q+1,good_pix] )
                    elif u==1 and q==0: Ncov[f,1::2,::2] += np.outer( noise_maps_sim[2*f+u+1,good_pix], noise_maps_sim[2*f+q+1,good_pix] )
                    else: Ncov[f,1::2,1::2] += np.outer( noise_maps_sim[2*f+u+1,good_pix], noise_maps_sim[2*f+q+1,good_pix] )
    return Ncov


def great_circle_distance(coord1, coord2):

    return np.arccos( np.sin(coord1[1])*np.sin(coord2[1]) + np.cos(coord1[1])*np.cos(coord2[1])*np.cos(coord2[0]-coord1[0]) )




def noise_correlation_estimation(self, binary_mask):
    from scipy.special import legendre
    from . import V3calc as v3

    costheta_v = np.linspace(-1,1,num=1000)
    theta_v = np.arccos(costheta_v)
   
    nh = hp.ud_grade(hp.read_map(self.get_input('norm_hits_map')), nside_out=self.config['nside'])
    msk = mknm.get_mask(nh, nside_out=self.config['nside'])
    fsky = np.mean(msk)

    ## grab the noise angular power spectra
    print('estimating N_ell')
    ll, nll, nlev=v3.so_V3_SA_noise(self.config['sensitivity_mode'], \
                        self.config['knee_mode'], self.config['ny_lf'], fsky, \
                        self.config['nside'])

    ## estimate the correlation noise function and interpolation
    print('estimate the correlation noise function and interpolation')
    from scipy.interpolate import interp1d
    Nfreqs = len(nll)
    Ntheta = np.zeros((Nfreqs, len(theta_v)))
    ell_v = range(len(nll[0])+2)
    Ntheta_interp = []
    for f in range(Nfreqs):
        print('f = ', f)
        for i_ct in range(len(costheta_v)):
            Ntheta[f, i_ct] = np.sum([(2.0*l + 1)/(4*np.pi) * nll[f][l-2] * legendre(l)(costheta_v[i_ct]) for l in ell_v[2:]])
        Ntheta_interp.append( interp1d(theta_v, Ntheta[f,:]) )

    ## assignment to pixels! 
    print('building N_ij')
    obs_pix = np.where(binary_mask == 1) [0]
    Nij = np.zeros((Nfreqs, len(obs_pix),len(obs_pix)))
    for f in range(Nfreqs):
        ind1=0
        print('progress = ', f)
        for p1 in obs_pix:
            ind2=0
            for p2 in obs_pix:
                # longlatp1,longlatp2 = hp.pix2ang(self.config['nside'], [p1, p2])
                longlatp1 = hp.pix2ang(self.config['nside'], p1)
                longlatp2 = hp.pix2ang(self.config['nside'], p2)
                theta_p1_p2 = np.abs(great_circle_distance(longlatp1, longlatp2))
                # if p1 == p2: 
                #     print(longlatp1, longlatp2)
                #     print(theta_p1_p2)
                #     exit()
                Nij[f, ind1, ind2] = Ntheta_interp[f](theta_p1_p2)
                ind2+=1
            ind1+=1

    return Nij

def noise_covariance_correction(cov_in, instrument, common_beam, nside_in, nside_out, Nsims):

    if common_beam == 0: 
        return cov_in

    Bl_gauss_common = hp.gauss_beam( np.radians(common_beam/60.0), lmax=2*nside_out)    
    ratio_av = np.zeros(len(instrument['frequency']))

    Nsims_loc = 10
    for i_sim in range(Nsims_loc):
        noise_p=np.random.normal(size=((len(instrument['frequency']), 3, 12*nside_in**2)))
        # noise_p=np.random.normal(size=((len(instrument['frequency']), 3, 12*nside_out**2)))
        sigma_p=instrument['depth_p']/hp.nside2resol(nside_in, arcmin=True)
        N_p=np.diag(sigma_p**2)
        L_p=scipy.linalg.sqrtm(N_p)
        noise_p=(L_p.dot(noise_p[np.newaxis].T)).T[0]
        noise_p = noise_p.swapaxes(-1,0)
        noise_p = noise_p.swapaxes(1,2)

        noise_p_beam_ = np.zeros((noise_p.shape[0],noise_p.shape[1], 12*nside_out**2))
        for f in range(noise_p.shape[0]):
            # if nside_out!=nside_in: noise_p_loc = hp.ud_grade(noise_p[f], nside_out=nside_out)
            # else: 
            noise_p_loc = noise_p[f]*1.0
            Bl_gauss_fwhm = hp.gauss_beam( np.radians(instrument['fwhm'][f]/60.0), lmax=2*nside_out)
            alms = hp.map2alm(noise_p_loc, lmax=3*nside_out)
            for alm_ in alms:
                hp.almxfl(alm_, Bl_gauss_common/Bl_gauss_fwhm, inplace=True)             
            noise_p_beam_[f] = hp.alm2map(alms, nside_out)
        noise_p = noise_p_beam_*1.0

        for f in range(noise_p.shape[0]):
            Q_std =  np.std(noise_p[f,1])*hp.nside2resol(nside_out, arcmin=True)
            U_std =  np.std(noise_p[f,2])*hp.nside2resol(nside_out, arcmin=True)
            print('white noise level = ', instrument['depth_p'][f],\
                     '// ratios -> ', round(instrument['depth_p'][f]/Q_std,4), ' / ', round(instrument['depth_p'][f]/U_std,4))
            ratio_av[f] += (instrument['depth_p'][f]/Q_std+instrument['depth_p'][f]/U_std)/2.0
    
    # ratio is INPUT/OUTPUT
    ratio_av /= Nsims_loc
    print('ratio uK-arcmin INPUT/OUTPUT = ',  ratio_av)

    cov_out = np.zeros_like(cov_in)
    for f in range(noise_p.shape[0]):
        cov_out[3*f:3*(f+1)] = cov_in[3*f:3*(f+1)] / ratio_av[f]**2

    return cov_out


class BBMapSim(PipelineStage):
    """
    Stage that performs the simulation 
    """
    name='BBMapSim'
    inputs= [('binary_mask',FitsFile),('norm_hits_map', FitsFile),('Cl_BB_prim_r1', FitsFile),('Cl_BB_lens', FitsFile)]
    outputs=[('binary_mask_cut',FitsFile),('frequency_maps',FitsFile),('noise_cov',FitsFile),('noise_maps',FitsFile),\
            ('CMB_template_150GHz',FitsFile),('dust_template_150GHz',FitsFile),('sync_template_150GHz',FitsFile),
            ('freq_maps_unbeamed', FitsFile), ('instrument', NumpyFile), ('noise_cov_beamed',FitsFile)]

    def run(self) :

        nh = hp.read_map(self.get_input('norm_hits_map'))
        nhits, noise_maps, nlev, nll = mknm.get_noise_sim(sensitivity=self.config['sensitivity_mode'], 
                        knee_mode=self.config['knee_mode'],ny_lf=self.config['ny_lf'],
                            nside_out=self.config['nside'], norm_hits_map=nh,
                                no_inh=self.config['no_inh'], CMBS4=self.config['instrument'])

        if self.config['external_binary_mask']:
            binary_mask = hp.read_map(self.config['external_binary_mask'])
            binary_mask = hp.ud_grade(binary_mask, nside_out=self.config['nside'])
        else:
            binary_mask = hp.read_map(self.get_input('binary_mask'))
            binary_mask = hp.ud_grade(binary_mask, nside_out=self.config['nside'])
            binary_mask[np.where(nhits<1e-6)[0]] = 0.0

        # GENERATE CMB AND FOREGROUNDS
        # d_config = models(self.config['dust_model'], self.config['nside'])
        # s_config = models(self.config['sync_model'], self.config['nside'])
        # c_config = models(self.config['cmb_model'], self.config['nside'])
        d_config = self.config['dust_model']
        s_config = self.config['sync_model']
        c_config = self.config['cmb_model']

        # performing the CMB simulation with synfast
        if self.config['cmb_sim_no_pysm']:
            Cl_BB_prim = self.config['r_input']*hp.read_cl(self.get_input('Cl_BB_prim_r1'))[2]
            Cl_lens = hp.read_cl(self.get_input('Cl_BB_lens'))
            l_max_lens = len(Cl_lens[0])
            Cl_BB_lens = self.config['A_lens']*Cl_lens[2]#[:l_max_prim]
            Cl_TT = Cl_lens[0]#[:l_max_prim]
            Cl_EE = Cl_lens[1]#[:l_max_prim]
            Cl_TE = Cl_lens[3]#[:l_max_prim]
            # sky_config = {'cmb' : '', 'dust' : d_config, 'synchrotron' : s_config}
            # sky = pysm.Sky(sky_config)
            # sky = pysm.Sky(nside=self.config['nside'], preset_strings=[self.config['dust_model'], self.config['sync_model']])
            sky = get_sky(self.config['nside'], d_config+s_config)
            Cl_BB = Cl_BB_prim[:l_max_lens] + Cl_BB_lens
            cmb_sky = hp.synfast([Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE*0.0, Cl_EE*0.0], nside=self.config['nside'], new=True)
        else:
            # sky_config = {'cmb' : c_config, 'dust' : d_config, 'synchrotron' : s_config}
            # sky = pysm.Sky(sky_config)
            # sky = pysm.Sky(nside=self.config['nside'], preset_strings=[self.config['cmb_model'], self.config['dust_model'], self.config['sync_model']])
            sky = get_sky(self.config['nside'], c_config+d_config+s_config)   

        # sky_config_CMB = {'cmb' : c_config}
        # sky_CMB = pysm.Sky(sky_config_CMB)
        # sky_CMB = pysm.Sky(nside=self.config['nside'], preset_strings=[self.config['cmb_model']])
        sky_CMB = get_sky(self.config['nside'], c_config)   
        # sky_config_dust = {'dust' : d_config}
        # sky_dust = pysm.Sky(sky_config_dust)
        # sky_dust = pysm.Sky(nside=self.config['nside'], preset_strings=[self.config['dust_model']])
        sky_dust = get_sky(self.config['nside'], d_config)   
        # sky_config_sync = {'synchrotron' : s_config}
        # sky_sync = pysm.Sky(sky_config_sync)
        # sky_sync = pysm.Sky(nside=self.config['nside'], preset_strings=[self.config['sync_model']])
        sky_sync = get_sky(self.config['nside'], s_config)   

        # DEFINE INSTRUMENT AND SCAN SKY
        if self.config['instrument'] == 'SO':
            fwhm = V3.so_V3_SA_beams()
            freqs = V3.so_V3_SA_bands()
        elif self.config['instrument'] == 'CMBS4':
            fwhm = np.array([11, 72.8, 25.5, 22.7, 25.5, 13, 10])
            # freqs = np.array([20, 30, 40, 85, 95, 145, 155, 220, 270])
            freqs = np.array([20, 30, 40, 90, 150, 220, 270])
            # nlev = np.array([6.09, 2.44, 3.09, 0.61, 0.54, 0.85, 0.91, 2.34, 4.02])
            # nlev = np.array([5.07, 4.20, 5.31, 6.40, 5.64, 4.21, 4.51, 34.59, 59.32])/3
            # nlev = np.array([5.52, 4.56, 5.78, 6.96, 6.14, 4.31, 4.61, 35.62, 61.08])/3
            nlev = np.array([9.42, 3.94, 4.98, 0.56, 0.90, 3.89, 6.67])
        else:
            print('I do not know this instrument')
            sys.exit()

        channels = []
        for f_ in freqs:
            channels.append((np.array([f_-1,f_,f_+1]),np.array([0.0, 1.0, 0.0])))
        print('channels = ', channels)

        instrument_config = {
            'nside' : self.config['nside'],
            'frequency' : freqs, 
            'use_smoothing' : False,
            'fwhm' : fwhm, 
            'add_noise' : False,
            'depth_i' : nlev/np.sqrt(2),
            'depth_p' : nlev,
            'noise_seed' : 1234,
            'use_bandpass' : False,
            'channels': channels,
            'channel_names': [str(f_) for f_ in freqs],
            'output_units' : 'uK_CMB',
            'output_directory' : './',
            'output_prefix' : self.config['tag'],
            }

        ###################################

        # instrument = pysm.Instrument(instrument_config)
        instrument_config_150GHz = copy.deepcopy(instrument_config)
        instrument_config_150GHz['frequency'] = np.array([150.0])
        instrument_config_150GHz['depth_i'] = np.array([1.0])
        instrument_config_150GHz['depth_p'] = np.array([1.0])
        instrument_150GHz = standardize_instrument(instrument_config_150GHz)

        # instrument.observe(sky)
        # freq_maps = instrument.observe(sky, write_outputs=False)[0]
        instrument = standardize_instrument(instrument_config)
        freq_maps = get_observation(instrument, sky) 

        if self.config['cmb_sim_no_pysm']:
            # adding CMB in this case
            for i in range(freq_maps.shape[0]):
                freq_maps[i,:,:] += cmb_sky[:,:]
            CMB_template_150GHz = cmb_sky
        else:
            # CMB_template_150GHz = instrument_150GHz.observe(sky_CMB, write_outputs=False)[0].reshape((3,noise_maps.shape[1]))
            CMB_template_150GHz = get_observation(instrument_150GHz, sky_CMB).reshape((3,noise_maps.shape[1]))
        # dust_template_150GHz = instrument_150GHz.observe(sky_dust, write_outputs=False)[0].reshape((3,noise_maps.shape[1]))
        dust_template_150GHz = get_observation(instrument_150GHz, sky_dust).reshape((3,noise_maps.shape[1]))
        # sync_template_150GHz = instrument_150GHz.observe(sky_sync, write_outputs=False)[0].reshape((3,noise_maps.shape[1]))
        sync_template_150GHz = get_observation(instrument_150GHz, sky_sync).reshape((3,noise_maps.shape[1]))

        # restructuration of the freq maps, of size {n_stokes x n_freqs, n_pix}
        # print('shape of freq_maps = ', freq_maps.shape)
        freq_maps = freq_maps.reshape((3*len(self.config['frequencies']),hp.nside2npix(self.config['nside'])))
        # print('shape of freq_maps = ', freq_maps.shape)
        NSIDE_INPUT_MAP = hp.npix2nside(len(freq_maps[0]))

        if self.config['combined_directory']!='':
            freq_maps *= 0.0
            print('LOADING EXTERNAL SKY-ONLY MAPS')
            for f in range(len(instrument.frequency)):
                print('loading combined foregrounds map for frequency ', str(int(instrument.frequency[f])))
                # freq_maps[3*f:3*(f+1),:] = hp.ud_grade(hp.read_map(list_of_files[f], field=None), nside_out=self.config['nside'])
                loc_freq_map = hp.read_map(glob.glob(os.path.join(self.config['combined_directory'],'SO_SAT_'+str(int(instrument.frequency[f]))+'_comb_*.fits'))[0], field=None)
                NSIDE_INPUT_MAP = hp.npix2nside(len(loc_freq_map[0]))
                alms = hp.map2alm(loc_freq_map, lmax=3*self.config['nside'])
                Bl_gauss_pix = hp.gauss_beam( hp.nside2resol(self.config['nside']), lmax=2*self.config['nside'])        
                for alm_ in alms: hp.almxfl(alm_, Bl_gauss_pix, inplace=True)             
                freq_maps[3*f:3*(f+1),:] = hp.alm2map(alms, self.config['nside'])
                # freq_maps[3*f:3*(f+1),:] = hp.ud_grade(loc_freq_map, nside_out=self.config['nside'])
                del loc_freq_map

            print('f=', f, ' freq_maps = ', freq_maps[3*f:3*(f+1),:])

        # adding noise
        if self.config['noise_option']=='no_noise': 
            pass
        elif self.config['external_noise_sims']!='' or self.config['Nico_noise_combination']:
            noise_maps = freq_maps*0.0
            # print('noise_maps.shape = ', noise_maps.shape)
            print('LOADING EXTERNAL NOISE-ONLY MAPS')

            if self.config['Nico_noise_combination']:
                if self.config['knee_mode'] == 2 : knee_mode_loc = None
                else: knee_mode_loc = self.config['knee_mode']
                factors = compute_noise_factors(self.config['sensitivity_mode'], knee_mode_loc)

            for f in range(len(instrument.frequency)):
                print('loading noise map for frequency ', str(int(instrument.frequency[f])))
                # noise_maps[3*f:3*(f+1),:] = hp.ud_grade(hp.read_map(list_of_files[f], field=None), nside_out=self.config['nside'])

                if self.config['Nico_noise_combination']:
                    noise_loc = combine_noise_maps(self.config['isim'], instrument.frequency[f], factors)
                else:
                    noise_loc = hp.read_map(glob.glob(os.path.join(self.config['external_noise_sims'],'SO_SAT_'+str(int(instrument.frequency[f]))+'_noise_FULL_*_white_20201207.fits'))[0], field=None)
                # noise_maps[3*f:3*(f+1),:] = hp.ud_grade(noise_loc, nside_out=self.config['nside'])
                alms = hp.map2alm(noise_loc, lmax=3*self.config['nside'])
                Bl_gauss_pix = hp.gauss_beam( hp.nside2resol(self.config['nside']), lmax=2*self.config['nside'])        
                for alm_ in alms: hp.almxfl(alm_, Bl_gauss_pix, inplace=True)             
                noise_maps[3*f:3*(f+1),:] = hp.alm2map(alms, self.config['nside'])  

                if ((not self.config['no_inh']) and (self.config['Nico_noise_combination'])):
                    # renormalize the noise map to take into account the effect of inhomogeneous noise
                    print('rescaling the noise maps with hits map')
                    # noise_loc /= np.sqrt(nh/np.max(nh))
                    nhits_nz = np.where(nhits!=0)[0]
                    noise_maps[3*f:3*(f+1),nhits_nz] /= np.sqrt(nhits[nhits_nz]/np.max(nhits[nhits_nz]))

                # print('f=', f, ' NOISE ', noise_maps[3*f:3*(f+1),:])

            freq_maps += noise_maps*binary_mask
        elif self.config['noise_option']=='white_noise':
            nlev_map = freq_maps*0.0
            for i in range(len(instrument.frequency)):
                nlev_map[3*i:3*i+3,:] = np.array([instrument.depth_i[i], instrument.depth_p[i], instrument.depth_p[i]])[:,np.newaxis]*np.ones((3,freq_maps.shape[-1]))
            # nlev_map = np.vstack(([instrument_config['sens_I'], instrument_config['sens_P'], instrument_config['sens_P']]))
            nlev_map /= hp.nside2resol(self.config['nside'], arcmin=True)
            noise_maps = np.random.normal(freq_maps*0.0, nlev_map, freq_maps.shape)
            freq_maps += noise_maps*binary_mask
        else: 
            freq_maps += noise_maps*binary_mask

        freq_maps_unbeamed = freq_maps*1.0
        noise_maps_beamed = noise_maps*1.0

        if self.config['common_beam_correction']!=0.0:
            print('  -> common beam correction: correcting for frequency-dependent beams and convolving with a common beam')
            Bl_gauss_common = hp.gauss_beam( np.radians(self.config['common_beam_correction']/60), lmax=2*self.config['nside'])        
            for f in range(len(instrument.frequency)):
                Bl_gauss_fwhm = hp.gauss_beam( np.radians(instrument.fwhm[f]/60), lmax=2*self.config['nside'])
                alms = hp.map2alm(freq_maps[3*f:3*(f+1),:], lmax=3*self.config['nside'])
                for alm_ in alms:
                    hp.almxfl(alm_, Bl_gauss_common/Bl_gauss_fwhm, inplace=True)             
                freq_maps[3*f:3*(f+1),:] = hp.alm2map(alms, self.config['nside'])   

                print('f=', f, ' freq_maps = ', freq_maps[3*f:3*(f+1),:])

                # alms_n = hp.map2alm(noise_maps_beamed[3*f:3*(f+1),:], lmax=3*self.config['nside'])
                # for alms_ in alms_n:
                #     hp.almxfl(alms_, Bl_gauss_common/Bl_gauss_fwhm, inplace=True)             
                # noise_maps_beamed[3*f:3*(f+1),:] = hp.alm2map(alms_n, self.config['nside'])   

                # should do it for the noise too
                # alms_n = hp.map2alm(noise_maps[3*f:3*(f+1),:], lmax=3*self.config['nside'])
                # for alm_n in alms_n:
                #     hp.almxfl(alm_n, Bl_gauss_common/Bl_gauss_fwhm, inplace=True)             
                # noise_maps[3*f:3*(f+1),:] = hp.alm2map(alms_n, self.config['nside'])

                # Bl_gauss_fwhm = hp.gauss_beam( np.radians(instrument_150GHz.fwhm[0]/60), lmax=2*self.config['nside'])
                # alms = hp.map2alm(CMB_template_150GHz, lmax=3*self.config['nside'])
                # for alm_ in alms:
                #     hp.almxfl(alm_, Bl_gauss_common/Bl_gauss_fwhm, inplace=True)             
                # CMB_template_150GHz = hp.alm2map(alms, self.config['nside'])   

        freq_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
        freq_maps_unbeamed[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
        noise_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
        # noise_maps_beamed[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
        CMB_template_150GHz[:,np.where(binary_mask==0)[0]] = hp.UNSEEN

        # noise covariance 
        if self.config['external_noise_cov']:
            print('/// EXT NOISE COV')
            noise_cov = hp.read_map(self.config['external_noise_cov'], field=None)
            noise_cov_beamed = noise_cov*1.0
        elif self.config['bypass_noise_cov']:
            print('/// BYPASS NOISE COV')
            tag = ''
            for f in instrument.frequency: tag += str(f)+'_'
            for key in ['common_beam_correction', 'no_inh', 'Nico_noise_combination', 'Nsims_bias', 'nside', 'sensitivity_mode', 'knee_mode']:
                tag += key+'_'+str(self.config[key])
            path_to_noise_cov = os.path.join('/global/cscratch1/sd/josquin/bypass_noise_cov_'+tag)
            if not os.path.exists(path_to_noise_cov+'.npy'):
                print('noise covariance is not on disk yet. Computing it now.')
                noise_cov, noise_cov_beamed = noise_covariance_estimation(self, freq_maps.shape, instrument, nhits)
                np.save(path_to_noise_cov, (noise_cov, noise_cov_beamed), allow_pickle=True)
            else: 
                noise_cov, noise_cov_beamed = np.load(path_to_noise_cov+'.npy')
        else:
            print('/// WHITE NOISE COV')
            noise_cov = freq_maps*0.0
            # nlev /= hp.nside2resol(self.config['nside'], arcmin=True)
            noise_cov[::3,:] = nlev[:,np.newaxis]/np.sqrt(2.0)
            noise_cov[1::3,:] = nlev[:,np.newaxis]
            noise_cov[2::3,:] = nlev[:,np.newaxis]
            noise_cov *= binary_mask
            # divind by the pixel size in arcmin
            noise_cov /=  hp.nside2resol(self.config['nside'], arcmin=True)
            if self.config['noise_option']!='white_noise' and self.config['noise_option']!='no_noise':
                noise_cov /= np.sqrt(nhits/np.amax(nhits))
            # we put it to square !
            noise_cov *= noise_cov
            noise_cov_beamed = noise_cov*1.0

        # if self.config['noise_cov_beam_correction']:
        if ((self.config['common_beam_correction']!=0.0) and (not self.config['bypass_noise_cov'])):
            print('/////////// noise_cov_beam_correction after beam convolution ///////////////')
            noise_cov_beamed = noise_covariance_correction(cov_in=noise_cov, instrument=instrument_config, 
                            common_beam=self.config['common_beam_correction'], nside_in=NSIDE_INPUT_MAP, 
                                nside_out=self.config['nside'], Nsims=self.config['Nsims_bias'])
        # else: noise_cov_beamed = noise_cov*1.0


        noise_cov[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
        noise_cov_beamed[:,np.where(binary_mask==0)[0]] = hp.UNSEEN

        if self.config['pixel_based_noise_cov']:
            noise_cov_pp_v2 = pp_noise_correlation_estimation(self, binary_mask)
            np.save('noise_cov_pp_v2', noise_cov_pp_v2)

            noise_cov_pp = pp_noise_covariance_estimation(self, binary_mask)
            np.save('noise_cov_pp', noise_cov_pp)

        # save on disk frequency maps, noise maps, noise_cov, binary_mask
        column_names = []
        [ column_names.extend( ('I_'+str(ch)+'GHz','Q_'+str(ch)+'GHz','U_'+str(ch)+'GHz')) for ch in freqs]

        hp.write_map(self.get_output('binary_mask_cut'), binary_mask, overwrite=True)
        hp.write_map(self.get_output('frequency_maps'), freq_maps, overwrite=True, column_names=column_names)
        hp.write_map(self.get_output('noise_cov'), noise_cov, overwrite=True, column_names=column_names)
        hp.write_map(self.get_output('noise_cov_beamed'), noise_cov_beamed, overwrite=True, column_names=column_names)
        hp.write_map(self.get_output('noise_maps'), noise_maps, overwrite=True, column_names=column_names)
        hp.write_map(self.get_output('CMB_template_150GHz'), CMB_template_150GHz, overwrite=True)
        hp.write_map(self.get_output('dust_template_150GHz'), dust_template_150GHz, overwrite=True)
        hp.write_map(self.get_output('sync_template_150GHz'), sync_template_150GHz, overwrite=True)
        hp.write_map(self.get_output('freq_maps_unbeamed'), freq_maps_unbeamed, overwrite=True)
        np.save(self.get_output('instrument'), instrument)

        print(' >>> completed map simulator step')

if __name__ == '__main__':
    results = PipelineStage.main()
