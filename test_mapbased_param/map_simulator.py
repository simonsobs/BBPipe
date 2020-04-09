from bbpipe import PipelineStage
from .types import FitsFile, TextFile
import numpy as np
import matplotlib
# matplotlib.use('agg')
# matplotlib.use('tkagg')
import matplotlib.pyplot as pl
import pysm
from pysm.nominal import models
from . import mk_noise_map2 as mknm
from . import V3calc as V3
import healpy as hp
import copy
import glob


def noise_covariance_estimation(self, binary_mask):
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

    costheta_v = np.linspace(-1,1,num=100)
    theta_v = np.arccos(costheta_v)
   
    nh = hp.ud_grade(hp.read_map(self.get_input('norm_hits_map')), nside_out=self.config['nside'])
    msk=mknm.get_mask(nh, nside_out=self.config['nside'])
    fsky=np.mean(msk)

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
            Ntheta[f, i_ct] += np.sum([1/(4*np.pi)*(2*l + 1)*nll[f][l-2]*legendre(l)(costheta_v[i_ct]) for l in ell_v[2:]])
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
                longlatp1,longlatp2 = hp.pix2ang(self.config['nside'], [p1, p2])
                theta_p1_p2 = np.abs(great_circle_distance(longlatp1, longlatp2))
                Nij[f, ind1, ind2] = Ntheta_interp[f](theta_p1_p2)
                ind2+=1
            ind1+=1

    return Nij


class BBMapSim(PipelineStage):
    """
    Stage that performs the simulation 
    """
    name='BBMapSim'
    inputs= [('binary_mask',FitsFile),('norm_hits_map', FitsFile),('Cl_BB_prim_r1', FitsFile),('Cl_BB_lens', FitsFile)]
    outputs=[('binary_mask_cut',FitsFile),('frequency_maps',FitsFile),('noise_cov',FitsFile),('noise_maps',FitsFile),\
            ('CMB_template_150GHz',FitsFile),('dust_template_150GHz',FitsFile),('sync_template_150GHz',FitsFile)]

    def run(self) :

        nhits, noise_maps, nlev = mknm.get_noise_sim(sensitivity=self.config['sensitivity_mode'], 
                        knee_mode=self.config['knee_mode'],ny_lf=self.config['ny_lf'],
                            nside_out=self.config['nside'], norm_hits_map=hp.read_map(self.get_input('norm_hits_map')),
                                no_inh=self.config['no_inh'], CMBS4=self.config['instrument'])

        if self.config['external_binary_mask']:
            binary_mask = hp.read_map(self.config['external_binary_mask'])
            binary_mask = hp.ud_grade(binary_mask, nside_out=self.config['nside'])
        else:
            binary_mask = hp.read_map(self.get_input('binary_mask'))
            binary_mask = hp.ud_grade(binary_mask, nside_out=self.config['nside'])
            binary_mask[np.where(nhits<1e-6)[0]] = 0.0

        # GENERATE CMB AND FOREGROUNDS
        d_config = models(self.config['dust_model'], self.config['nside'])
        s_config = models(self.config['sync_model'], self.config['nside'])
        c_config = models(self.config['cmb_model'], self.config['nside'])
       
        # performing the CMB simulation with synfast
        if self.config['cmb_sim_no_pysm']:
            Cl_BB_prim = self.config['r_input']*hp.read_cl(self.get_input('Cl_BB_prim_r1'))[2]
            Cl_lens = hp.read_cl(self.get_input('Cl_BB_lens'))
            l_max_lens = len(Cl_lens[0])
            Cl_BB_lens = self.config['A_lens']*Cl_lens[2]#[:l_max_prim]
            Cl_TT = Cl_lens[0]#[:l_max_prim]
            Cl_EE = Cl_lens[1]#[:l_max_prim]
            Cl_TE = Cl_lens[3]#[:l_max_prim]
            sky_config = {'cmb' : '', 'dust' : d_config, 'synchrotron' : s_config}
            sky = pysm.Sky(sky_config)
            Cl_BB = Cl_BB_prim[:l_max_lens] + Cl_BB_lens
            cmb_sky = hp.synfast([Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE*0.0, Cl_EE*0.0], nside=self.config['nside'], new=True)
        else:
            sky_config = {'cmb' : c_config, 'dust' : d_config, 'synchrotron' : s_config}
            sky = pysm.Sky(sky_config)
      
        sky_config_CMB = {'cmb' : c_config}
        sky_CMB = pysm.Sky(sky_config_CMB)
        sky_config_dust = {'dust' : d_config}
        sky_dust = pysm.Sky(sky_config_dust)
        sky_config_sync = {'synchrotron' : s_config}
        sky_sync = pysm.Sky(sky_config_sync)

        # DEFINE INSTRUMENT AND SCAN SKY
        if self.config['instrument'] == 'SO':
            fwhm = V3.so_V3_SA_beams()
            freqs = V3.so_V3_SA_bands()
        elif self.config['instrument'] == 'CMBS4':
            fwhm = np.array([11, 72.8, 72.8, 25.5, 22.7, 25.5, 22.7, 13, 13])
            freqs = np.array([20, 30, 40, 85, 95, 145, 155, 220, 270])
            # nlev = np.array([6.09, 2.44, 3.09, 0.61, 0.54, 0.85, 0.91, 2.34, 4.02])
            # nlev = np.array([5.07, 4.20, 5.31, 6.40, 5.64, 4.21, 4.51, 34.59, 59.32])/3
            nlev = np.array([5.52, 4.56, 5.78, 6.96, 6.14, 4.31, 4.61, 35.62, 61.08])/3
        else:
            print('I do not know this instrument')
            sys.exit()

        channels = []
        for f_ in freqs:
            channels.append((np.array([f_-1,f_,f_+1]),np.array([0.0, 1.0, 0.0])))
        print('channels = ', channels)

        instrument_config = {
            'nside' : self.config['nside'],
            'frequencies' : freqs, 
            'use_smoothing' : False,
            'beams' : fwhm, 
            'add_noise' : False,
            'sens_I' : nlev/np.sqrt(2),
            'sens_P' : nlev,
            'noise_seed' : 1234,
            'use_bandpass' : False,
            'channels': channels,
            'channel_names': [str(f_) for f_ in freqs],
            'output_units' : 'uK_CMB',
            'output_directory' : './',
            'output_prefix' : self.config['tag'],
            }

        instrument = pysm.Instrument(instrument_config)
        instrument_config_150GHz = copy.deepcopy(instrument_config)
        instrument_config_150GHz['frequencies'] = np.array([150.0])
        instrument_config_150GHz['sens_I'] = np.array([1.0])
        instrument_config_150GHz['sens_P'] = np.array([1.0])
        instrument_150GHz = pysm.Instrument(instrument_config_150GHz)

        # instrument.observe(sky)
        freq_maps = instrument.observe(sky, write_outputs=False)[0]
        if self.config['cmb_sim_no_pysm']:
            # adding CMB in this case
            for i in range(freq_maps.shape[0]):
                freq_maps[i,:,:] += cmb_sky[:,:]
            CMB_template_150GHz = cmb_sky
        else:
            CMB_template_150GHz = instrument_150GHz.observe(sky_CMB, write_outputs=False)[0].reshape((3,noise_maps.shape[1]))
        dust_template_150GHz = instrument_150GHz.observe(sky_dust, write_outputs=False)[0].reshape((3,noise_maps.shape[1]))
        sync_template_150GHz = instrument_150GHz.observe(sky_sync, write_outputs=False)[0].reshape((3,noise_maps.shape[1]))

        # restructuration of the freq maps, of size {n_stokes x n_freqs, n_pix}
        # print('shape of freq_maps = ', freq_maps.shape)
        freq_maps = freq_maps.reshape((3*len(self.config['frequencies']),hp.nside2npix(self.config['nside'])))
        # print('shape of freq_maps = ', freq_maps.shape)

        if self.config['external_sky_sims']!='':
            freq_maps *= 0.0
            print('freq_maps.shape = ', freq_maps.shape)
            print('EXTERNAL SKY-ONLY MAPS LOADED')
            list_of_files = sorted(glob.glob(self.config['external_sky_sims']))   
            print('       -> list_of_files :', list_of_files)
            for f in range(len(list_of_files)):
                print('loading ... ', list_of_files[f])
                freq_maps[3*f:3*(f+1),:] = hp.ud_grade(hp.read_map(list_of_files[f], field=None), nside_out=self.config['nside'])
        # adding noise
        if self.config['noise_option']=='white_noise':
            nlev_map = freq_maps*0.0
            for i in range(len(instrument_config['frequencies'])):
                nlev_map[3*i:3*i+3,:] = np.array([instrument_config['sens_I'][i], instrument_config['sens_P'][i], instrument_config['sens_P'][i]])[:,np.newaxis]*np.ones((3,freq_maps.shape[-1]))
            # nlev_map = np.vstack(([instrument_config['sens_I'], instrument_config['sens_P'], instrument_config['sens_P']]))
            nlev_map /= hp.nside2resol(self.config['nside'], arcmin=True)
            noise_maps = np.random.normal(freq_maps*0.0, nlev_map, freq_maps.shape)*binary_mask
            freq_maps += noise_maps*binary_mask
        elif self.config['noise_option']=='no_noise': 
            pass
        elif self.config['external_noise_sims']!='':
            noise_maps = freq_maps*0.0
            print('noise_maps.shape = ', noise_maps.shape)
            print('EXTERNAL NOISE-ONLY MAPS LOADED')
            list_of_files = sorted(glob.glob(self.config['external_noise_sims']))   
            print('       -> list_of_files :', list_of_files)
            for f in range(len(list_of_files)):
                noise_maps[3*f:3*(f+1),:] = hp.ud_grade(hp.read_map(list_of_files[f], field=None), nside_out=self.config['nside'])
            freq_maps += noise_maps*binary_mask
        else: 
            freq_maps += noise_maps*binary_mask

        freq_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
        noise_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN

        # noise covariance 
        if self.config['external_noise_cov']:
            noise_cov = hp.read_map(self.config['external_noise_cov'], field=None)
        else:
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

        noise_cov[:,np.where(binary_mask==0)[0]] = hp.UNSEEN

        if self.config['pixel_based_noise_cov']:
            noise_cov_pp_v2 = noise_correlation_estimation(self, binary_mask)
            np.save('noise_cov_pp_v2', noise_cov_pp_v2)

            noise_cov_pp = noise_covariance_estimation(self, binary_mask)
            np.save('noise_cov_pp', noise_cov_pp)


        # save on disk frequency maps, noise maps, noise_cov, binary_mask
        column_names = []
        [ column_names.extend( ('I_'+str(ch)+'GHz','Q_'+str(ch)+'GHz','U_'+str(ch)+'GHz')) for ch in freqs]

        hp.write_map(self.get_output('binary_mask_cut'), binary_mask, overwrite=True)
        hp.write_map(self.get_output('frequency_maps'), freq_maps, overwrite=True, column_names=column_names)
        hp.write_map(self.get_output('noise_cov'), noise_cov, overwrite=True, column_names=column_names)
        hp.write_map(self.get_output('noise_maps'), noise_maps, overwrite=True, column_names=column_names)
        hp.write_map(self.get_output('CMB_template_150GHz'), CMB_template_150GHz, overwrite=True)
        hp.write_map(self.get_output('dust_template_150GHz'), dust_template_150GHz, overwrite=True)
        hp.write_map(self.get_output('sync_template_150GHz'), sync_template_150GHz, overwrite=True)

if __name__ == '__main__':
    results = PipelineStage.main()
