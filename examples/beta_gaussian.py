import pysm
from pysm.nominal import models
import healpy as hp
import numpy as np

nside = 256 
lmax = 3*nside-1
ells = np.arange(lmax+1)

# Gamma (convergence)
gamma_beta_sync = -2.5 
gamma_beta_dust = -3.5

# Calculate power spectrum (input)
def powerlaw(ells, gamma):
    c_ells = ((ells+0.00000000000001) / 80.)**gamma
    c_ells[ells<1]=c_ells[1]
    return c_ells

cl_betaSync = powerlaw(ells, gamma_beta_sync)
cl_betaDust = powerlaw(ells, gamma_beta_dust)

# Map from given power spectrum as mean + gaussian random field
def map_beta(ells, gamma, mean_add, sigma_des=0.1):
    cl_beta = powerlaw(ells, gamma)
    delta_map = hp.synfast(cl_beta, nside, new=True, verbose=False)
    sigma_map = np.std(delta_map)
    delta_beta = delta_map * sigma_des / sigma_map
    map_return = delta_beta+mean_add
    print(np.std(map_return), np.mean(map_return))
    return map_return

map_beta_sync = map_beta(ells, gamma_beta_sync, -3.)
map_beta_dust = map_beta(ells, gamma_beta_dust, 1.6)

prefix_out = "."
hp.write_map(prefix_out+"/map_beta_sync.fits", map_beta_sync,  overwrite=True)
hp.write_map(prefix_out+"/map_beta_dust.fits", map_beta_dust, overwrite=True) 

map_beta_sync_fin = hp.synfast(cl_betaSync, nside, new=True, verbose=False) - map_beta_sync
map_beta_dust_fin = hp.synfast(cl_betaDust, nside, new=True, verbose=False) - map_beta_dust

# Power spectrum (output)
cl_betaSync_out = hp.anafast(map_beta_sync_fin, lmax=lmax)
cl_betaDust_out = hp.anafast(map_beta_dust_fin, lmax=lmax)
