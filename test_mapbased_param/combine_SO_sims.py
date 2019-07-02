# use Simons_Obs simulations
# path is using the following convention : 
# {output_nside}/{content}/{num:04d}/simonsobs_{content}_uKCMB_{telescope}{band:03d}_nside{nside}_{num:04d}.fits"

import os
import numpy as np
import healpy as hp
import glob

path_to_files = '/project/projectdirs/sobs/v4_sims/mbs/201901_gaussian_fg_lensed_cmb_realistic_noise/512'

CMB_loc = hp.read_map(os.path.join(path_to_files, 'cmb/0010/simonsobs_cmb_uKCMB_sa027_nside512_0010.fits'), field=None)

output_freq_maps = [np.zeros_like(CMB_loc)]*6
freqs = ['027', '039', '093', '145', '225', '280']
sims_ = '0010' # because there is a single foregrounds realization

indf = 0
for f in freqs:
	for s in ['cmb', 'dust', 'synchrotron']:
		output_freq_maps[indf] += hp.read_map(os.path.join(path_to_files, s+'/'+sims_+'/simonsobs_'+s+'_uKCMB_sa'+f+'_nside512_0010.fits'), field=None)

	print(output_freq_maps[indf].shape)
	hp.write_map(output_freq_maps[indf].T, './201901_gaussian_fg_lensed_cmb_uKCMB_sa'+f+'_nside512_'+sims_+'.fits')

exit()