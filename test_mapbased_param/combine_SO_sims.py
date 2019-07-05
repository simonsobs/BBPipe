# use Simons_Obs simulations
# path is using the following convention : 
# {output_nside}/{content}/{num:04d}/simonsobs_{content}_uKCMB_{telescope}{band:03d}_nside{nside}_{num:04d}.fits"

import os
import numpy as np
import healpy as hp
import glob
import pylab as pl

path_to_files = '/project/projectdirs/sobs/v4_sims/mbs/201901_gaussian_fg_lensed_cmb_realistic_noise/512'

CMB_loc = hp.read_map(os.path.join(path_to_files, 'cmb/0010/simonsobs_cmb_uKCMB_sa027_nside512_0010.fits'), field=None)

output_freq_maps = [np.zeros_like(CMB_loc)]*6
freqs = ['027', '039', '093', '145', '225', '280']
sims_ = '0010' # because there is a single foregrounds realization
freqs_ = [27, 39, 93, 145, 225, 280]
std_cmb = []
std_dust = []
std_sync = []
# components = ['cmb', 'dust', 'synchrotron']
components = ['dust', 'synchrotron']
indf = 0
for f in freqs:
	for s in components:
		output_freq_maps[indf] += hp.read_map(os.path.join(path_to_files, s+'/'+sims_+'/simonsobs_'+s+'_uKCMB_sa'+f+'_nside512_0010.fits'), field=None)
		print(output_freq_maps[indf].shape)
		if s == 'cmb': std_cmb.append(np.std(hp.read_map(os.path.join(path_to_files, s+'/'+sims_+'/simonsobs_'+s+'_uKCMB_sa'+f+'_nside512_0010.fits'), field=None)[1]))
		if s == 'dust': std_dust.append(np.std(hp.read_map(os.path.join(path_to_files, s+'/'+sims_+'/simonsobs_'+s+'_uKCMB_sa'+f+'_nside512_0010.fits'), field=None)[1]))
		if s == 'synchrotron': std_sync.append(np.std(hp.read_map(os.path.join(path_to_files, s+'/'+sims_+'/simonsobs_'+s+'_uKCMB_sa'+f+'_nside512_0010.fits'), field=None)[1]))
	hp.write_map('./201901_gaussian_fg_lensed_cmb_uKCMB_sa'+f+'_nside512_'+sims_+'.fits', output_freq_maps[indf], overwrite=True)
	
	indf+= 1


pl.figure()
# print(len(freqs_))
# print(len(std_cmb))
# print(len(std_dust))
# print(len(std_sync))
pl.figure()
# pl.plot(freqs_, std_cmb, 'k-', label='CMB')
pl.plot(freqs_, std_dust, 'r-', label='dust')
pl.plot(freqs_, std_sync, 'b-', label='sync')
pl.plot(freqs_, np.std(output_freq_maps[1::3], axis=1), 'k--', label='tot')
pl.legend()
pl.xlabel('frequency [GHz]')
pl.ylabel('standard deviation of the Q map [uK_CMB]')
pl.show()



exit()