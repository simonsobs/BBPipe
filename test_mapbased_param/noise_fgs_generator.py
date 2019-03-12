"""
Imhomogeneous noise
+ CMB + foregrounds 
SAT frequency maps
generator 
"""

import healpy as hp
import numpy as np
import pylab as pl
import argparse
import pysm
from pysm.nominal import models
import mk_noise_map as mknm
import V3calc as V3
import os.path as op

def grabargs():
	parser = argparse.ArgumentParser()
	parser.add_argument("--sensitivity_mode", type=int,
						help = "sensitivity mode for SATs",
						default=1)
	parser.add_argument("--knee_mode", type=int,
						help = "knee mode for SATs",
						default=1)
	parser.add_argument("--low_freq_year", type=float,
						help = "Number of years for the low frequency channels",
						default=1.0)	
	parser.add_argument("--nside", type=int,
						help = "nside for the output maps resolutions",
						default=128)	
	parser.add_argument("--output_directory", type=str, \
						help = "folder to output fits files", \
						default='.')
	parser.add_argument("--tag", type=str, \
						help = "specific tag for this run", \
						default='SO_sims')
	parser.add_argument('--white_noise', dest='white_noise', action='store_true', \
					help='add white_noise',\
					default=False)
	parser.add_argument('--no_noise', dest='no_noise', action='store_true', \
					help='do not add any noise',\
					default=False)
	args = parser.parse_args()
	return args


def main():

	args = grabargs()
	# GENERATE NOISE MAP
	nhits,noise_maps,nlev = mknm.get_noise_sim(sensitivity=args.sensitivity_mode, 
					knee_mode=args.knee_mode,ny_lf=args.low_freq_year,nside_out=args.nside)
	binary_mask = hp.read_map('mask_04000.fits')
	binary_mask = hp.ud_grade(binary_mask, nside_out=args.nside)
	binary_mask[np.where(nhits<1e-6)[0]] = 0.0
	# GENERATE CMB AND FOREGROUNDS
	d_config = models("d1", args.nside)
	s_config = models("s1", args.nside)
	c_config = models("c1", args.nside)
	sky_config = {'cmb' : c_config, 'dust' : d_config, 'synchrotron' : s_config}
	sky = pysm.Sky(sky_config)
	# DEFINE INSTRUMENT AND SCAN SKY
	fwhm = V3.so_V3_SA_beams()
	freqs = V3.so_V3_SA_bands()
	instrument_config = {
	    'nside' : args.nside,
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
	    'output_prefix' : args.tag,
		}

	instrument = pysm.Instrument(instrument_config)
	
	# instrument.observe(sky)
	freq_maps = instrument.observe(sky, write_outputs=False)[0]
	# restructuration of the noise map
	freq_maps = freq_maps.reshape(noise_maps.shape)
	# adding noise
	if args.white_noise:
		nlev_map = freq_maps*0.0
		for i in range(len(instrument_config['frequencies'])):
			nlev_map[3*i:3*i+3,:] = np.array([instrument_config['sens_I'][i], instrument_config['sens_P'][i], instrument_config['sens_P'][i]])[:,np.newaxis]*np.ones((3,freq_maps.shape[-1]))
		# nlev_map = np.vstack(([instrument_config['sens_I'], instrument_config['sens_P'], instrument_config['sens_P']]))
		nlev_map /= hp.nside2resol(args.nside, arcmin=True)
		noise_maps = np.random.normal(freq_maps*0.0, nlev_map, freq_maps.shape)*binary_mask
		freq_maps += noise_maps
	elif args.no_noise: pass
	else: freq_maps += noise_maps*binary_mask
	# freq_maps *= binary_mask
	# noise_maps *= binary_mask
	freq_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
	noise_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
	'''
	pl.figure()
	for i in range(len(instrument_config['frequencies'])):
	    hp.mollview( freq_maps[3*i,:], sub=(len(instrument_config['frequencies']),3,3*i+1), title='I '+str(instrument_config['frequencies'][i])+'GHz')
	    hp.mollview( freq_maps[3*i+1,:], sub=(len(instrument_config['frequencies']),3,3*i+2), title='Q '+str(instrument_config['frequencies'][i])+'GHz')
	    hp.mollview( freq_maps[3*i+2,:], sub=(len(instrument_config['frequencies']),3,3*i+3), title='U '+str(instrument_config['frequencies'][i])+'GHz')
	# pl.figure()
	# for i in range(len(instrument_config['frequencies'])):
	#     hp.mollview( noise_maps[3*i,:], sub=(len(instrument_config['frequencies']),3,3*i+1))
	#     hp.mollview( noise_maps[3*i+1,:], sub=(len(instrument_config['frequencies']),3,3*i+2))
	#     hp.mollview( noise_maps[3*i+2,:], sub=(len(instrument_config['frequencies']),3,3*i+3))
	pl.savefig('/Users/josquin1/Documents/Dropbox/CNRS-CR2/Simons_Observatory/BBPipe/test_mapbased_param/example_frequency_maps.pdf')
	pl.show()
	exit()
	'''

	# noise covariance 
	noise_cov = freq_maps*0.0
	noise_cov[::3,:] = nlev[:,np.newaxis]/np.sqrt(2.0)
	noise_cov[1::3,:] = nlev[:,np.newaxis]
	noise_cov[2::3,:] = nlev[:,np.newaxis]
	noise_cov *= binary_mask
	if not args.white_noise and not args.no_noise:
		noise_cov /= np.sqrt(nhits/np.amax(nhits))
	# we put it to square !
	noise_cov *= noise_cov
	# noise_cov *= binary_mask
	# noise_cov[:,np.where(binary_mask==0)[0]] = 1.0
	noise_cov[:,np.where(binary_mask==0)[0]] = hp.UNSEEN

	# save on disk frequency maps, noise maps, noise_cov, binary_mask
	tag = '_nside'+str(args.nside)
	tag += '_sens'+str(args.sensitivity_mode)
	tag += '_knee'+str(args.knee_mode)
	tag += '_nylf'+str(args.low_freq_year)
	if args.white_noise: tag += '_WN'

	column_names = []
	[ column_names.extend( ('I_'+str(ch)+'GHz','Q_'+str(ch)+'GHz','U_'+str(ch)+'GHz')) for ch in freqs]
	hp.write_map( op.join(args.output_directory, instrument_config['output_prefix']+'_binary_mask.fits'), binary_mask, overwrite=True)
	hp.write_map( op.join(args.output_directory, instrument_config['output_prefix']+'_noise_mask.fits'), nhits, overwrite=True)
	hp.write_map( op.join(args.output_directory, instrument_config['output_prefix']+'_frequency_maps'+tag+'.fits'), freq_maps, overwrite=True, column_names=column_names)
	hp.write_map( op.join(args.output_directory, instrument_config['output_prefix']+'_noise_cov'+tag+'.fits'), noise_cov, overwrite=True, column_names=column_names)
	hp.write_map( op.join(args.output_directory, instrument_config['output_prefix']+'_noise_maps'+tag+'.fits'), noise_maps, overwrite=True, column_names=column_names)

if __name__ == "__main__":

	main( )