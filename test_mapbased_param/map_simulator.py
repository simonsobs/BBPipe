from bbpipe import PipelineStage
from .types import FitsFile, TextFile
import numpy as np
import pylab as pl
import pysm
from pysm.nominal import models
from . import mk_noise_map2 as mknm
from . import V3calc as V3
import healpy as hp
import copy

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
                            nside_out=self.config['nside'], norm_hits_map=hp.read_map(self.get_input('norm_hits_map')))

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
            l_max_prim = len(Cl_BB_prim)
            Cl_lens = hp.read_cl(self.get_input('Cl_BB_lens'))
            Cl_BB_lens = self.config['A_lens']*Cl_lens[2][:l_max_prim]
            Cl_TT = Cl_lens[0][:l_max_prim]
            Cl_EE = Cl_lens[1][:l_max_prim]
            Cl_TE = Cl_lens[3][:l_max_prim]
            sky_config = {'cmb' : '', 'dust' : d_config, 'synchrotron' : s_config}
            sky = pysm.Sky(sky_config)
            Cl_BB = Cl_BB_prim + Cl_BB_lens
            cmb_sky = hp.synfast([Cl_TT, Cl_EE, Cl_BB, Cl_EE, Cl_EE*0.0, Cl_EE*0.0], nside=self.config['nside'])
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
        fwhm = V3.so_V3_SA_beams()
        freqs = V3.so_V3_SA_bands()
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

        # restructuration of the noise map
        freq_maps = freq_maps.reshape(noise_maps.shape)
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
        else: 
            freq_maps += noise_maps*binary_mask

        freq_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
        noise_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN

        # noise covariance 
        noise_cov = freq_maps*0.0
        noise_cov[::3,:] = nlev[:,np.newaxis]/np.sqrt(2.0)
        noise_cov[1::3,:] = nlev[:,np.newaxis]
        noise_cov[2::3,:] = nlev[:,np.newaxis]
        noise_cov *= binary_mask
        if self.config['noise_option']!='white_noise' and self.config['noise_option']!='no_noise':
            noise_cov /= np.sqrt(nhits/np.amax(nhits))
        # we put it to square !
        noise_cov *= noise_cov
        # noise_cov *= binary_mask
        # noise_cov[:,np.where(binary_mask==0)[0]] = 1.0
        noise_cov[:,np.where(binary_mask==0)[0]] = hp.UNSEEN

        # save on disk frequency maps, noise maps, noise_cov, binary_mask
        # tag = '_nside'+str(self.config['nside'])
        # tag += '_sens'+str(self.config['sensitivity_mode'])
        # tag += '_knee'+str(self.config['knee_mode'])
        # tag += '_nylf'+str(self.config['low_freq_year'])
        # if self.config['noise_option']=='white_noise': tag += '_white_noise'
        # if  self.config['noise_option']=='no_noise': tag += '_no_noise'

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
