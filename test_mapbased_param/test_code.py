import healpy as hp
import pylab as pl
import numpy as np
import os.path as op
# import pysm
# from fgbuster.pysm_helpers import get_instrument, get_sky
# import mk_noise_map as mknm

binary_mask=hp.read_map('/global/csratch1/sd/josquin/SO_sims/SO_sims_binary_mask.fits',verbose=False)
frequency_maps=hp.read_map('/global/csratch1/sd/josquin/SO_sims/SO_sims_frequency_maps_nside128_sens1_knee1_nylf1.0.fits',verbose=False, field=None)
# frequency_maps=hp.read_map('SO_sims_frequency_maps_nside128_sens1_knee1_nylf1.0_WN.fits',verbose=False, field=None)
noise_cov=hp.read_map('/global/csratch1/sd/josquin/SO_sims/SO_sims_noise_cov_nside128_sens1_knee1_nylf1.0.fits',verbose=False, field=None)
# noise_cov=hp.read_map('SO_sims_noise_cov_nside128_sens1_knee1_nylf1.0_WN.fits',verbose=False, field=None)
noise_maps=hp.read_map('/global/csratch1/sd/josquin/SO_sims/SO_sims_noise_maps_nside128_sens1_knee1_nylf1.0.fits',verbose=False, field=None)
# noise_maps=hp.read_map('SO_sims_noise_maps_nside128_sens1_knee1_nylf1.0_WN.fits',verbose=False, field=None)



# perform component separation
# assuming inhomogeneous noise
import fgbuster
from fgbuster.component_model import CMB, Dust, Synchrotron
components = [CMB(), Dust(150., temp=20.0), Synchrotron(150.)]

# nhits,nm_,nlev = mknm.get_noise_sim(sensitivity=1, 
                    # knee_mode=1,ny_lf=1,nside_out=128)

instrument = {'frequencies':np.array([30.0, 40.0, 95.0, 150.0, 220.0, 270.0])}#,
                # 'sens_P':nlev, 'sens_I':nlev/np.sqrt(2)}

from fgbuster.mixingmatrix import MixingMatrix

A = MixingMatrix(*components)
A_ev = A.evaluator(instrument['frequencies'])
A_dB_ev = A.diff_evaluator(instrument['frequencies'])
# print A_dB_ev(np.array([1.54, -3]))

# for i in range(frequency_maps.shape[0]):
#     masked_ind = np.where((frequency_maps[i,:]==hp.UNSEEN))[0]
#     frequency_maps[i,masked_ind] = 0.0
#     noise_maps[i,masked_ind] = 0.0

# frequency_maps -= noise_maps
# reorganization of maps
ind = 0
frequency_maps_ = np.zeros((len(instrument['frequencies']), 3, frequency_maps.shape[-1]))
noise_maps_ = np.zeros((len(instrument['frequencies']), 3, frequency_maps.shape[-1]))
noise_cov_ = np.zeros((len(instrument['frequencies']), 3, frequency_maps.shape[-1]))
for f in range(len(instrument['frequencies'])) : 
    for i in range(3): 
        frequency_maps_[f,i,:] =  frequency_maps[ind,:]*1.0
        noise_maps_[f,i,:] =  noise_maps[ind,:]*1.0
        noise_cov_[f,i,:] =  noise_cov[ind,:]*1.0
        ind += 1
# frequency_maps = frequency_maps.reshape((len(instrument['frequencies']), 3, frequency_maps.shape[-1]))
# print frequency_maps.shape
# removing I
frequency_maps_ = frequency_maps_[:,1:,:]
# noise_maps = noise_maps.reshape((len(instrument['frequencies']), 3, noise_maps.shape[-1]))
noise_maps_ = noise_maps_[:,1:,:]
noise_cov_ = noise_cov_[:,1:,:]
# for i in range(frequency_maps_.shape[0]):
#     hp.mollview( frequency_maps_[i,0,:], sub=(frequency_maps_.shape[0],2,2*i+1), min=-10,max=10)
#     hp.mollview( frequency_maps_[i,1,:], sub=(frequency_maps_.shape[0],2,2*i+2), min=-10,max=10)
# pl.figure()
# for i in range(frequency_maps_.shape[0]):
#     hp.mollview( noise_maps_[i,0,:], sub=(noise_maps_.shape[0],2,2*i+1), min=-10,max=10)
#     hp.mollview( noise_maps_[i,1,:], sub=(noise_maps_.shape[0],2,2*i+2), min=-10,max=10)
# pl.show()

# from fgbuster.separation_recipies import basic_comp_sep, weighted_comp_sep
# res = fgbuster.separation_recipies.basic_comp_sep(components, instrument,
#                             data=frequency_maps_ )
# print res

res = fgbuster.separation_recipies.weighted_comp_sep(components, instrument,
         data=frequency_maps_, cov=noise_cov_)

# saving results to disk now .... 
column_names = []
if res.s.shape[1] == 1:
    optI = 1
    optQU = 0
elif res.s.shape[1] == 2:
    optI = 0
    optQU = 1
else: 
    optI = 1
    optQU = 1

[ column_names.extend( ('I_'+str(ch)+'GHz'*optI,'Q_'+str(ch)+'GHz'*optQU,'U_'+str(ch)+'GHz'*optQU)) for ch in instrument['frequencies']]
# hp.write_map( op.join(args.output_directory, instrument_config['output_prefix']+'_binary_mask.fits'), binary_mask, overwrite=True)

# print res.s
# print res.s.shape
CMB_estimated=res.s[0,:,:]*1.0
# print '#################'
# print res.invAtNA
# print res.invAtNA.shape
CMB_noise_cov_estimated = res.invAtNA[0,0,:,:]*1.0
CMB_estimated_ = (1.0/CMB_noise_cov_estimated[0,0])*res.s[0,:,:]
CMB_estimated_[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
# pl.figure()
# hp.mollview(CMB_estimated[0], sub=(121))
# hp.mollview(CMB_estimated[1], sub=(122))
# pl.figure()
# hp.mollview(CMB_estimated_[0], sub=(121))
# hp.mollview(CMB_estimated_[1], sub=(122))
# pl.figure()
# hp.mollview(np.log10(CMB_noise_cov_estimated[0]), sub=(121))
# hp.mollview(np.log10(CMB_noise_cov_estimated[1]), sub=(122))
# pl.show()
# print '#################'
print(res.x)
# print res.x.shape
# print '#################'
print(res.Sigma)
# print np.diag(res.Sigma)


exit()