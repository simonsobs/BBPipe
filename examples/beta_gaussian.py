import healpy as hp
import numpy as np

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--seed', dest='seed',  default=1300, type=int,
                  help='Set to define seed, default=1300')
parser.add_option('--nside', dest='nside', default=256, type=int,
                  help='Set to define Nside parameter, default=256')
parser.add_option('--sigma-d', dest='sigma_dust', default=0,  type=int,
                  help='Modify amplitude of dust variation, default=1')
parser.add_option('--sigma-s', dest='sigma_sync', default=0,  type=int,
                  help='Modify amplitude of sync variation, default=1')
parser.add_option('--beta-dust', dest='calculate_beta_dust', default=True, action='store_true',
                  help='Calculate gaussian spectral index map of dust, default=True')
parser.add_option('--beta-sync', dest='calculate_beta_sync', default=True, action='store_true',
                  help='Calculate gaussian spectral index map of synchrotron, default=True')
(o, args) = parser.parse_args()

nside = o.nside
seed = o.seed
np.random.seed(seed)

lmax = 3*nside-1
ells = np.arange(lmax+1)

# Gamma (convergence)
gamma_beta_sync = -2.5 
gamma_beta_dust = -3.5
#gamma_beta_dust = -2.1

prefix_out="."

# Calculate power spectrum (input)
def powerlaw(ells, gamma):
    c_ells = ((ells+0.00000000000001) / 80.)**gamma
    c_ells[ells<2]=c_ells[2]
    return c_ells

if o.calculate_beta_dust:
    print(o.sigma_dust/100)
    cl_betaDust = powerlaw(ells, gamma_beta_dust)
    delta_mapD = hp.synfast(cl_betaDust, nside, new=True, verbose=False)
    sigma_mapD = np.std(delta_mapD)
    delta_betaD = delta_mapD * ((o.sigma_dust/100) / sigma_mapD)
    #print(delta_betaD)
    mean_addD = 1.6
    map_beta_dust = delta_betaD+mean_addD
    #print(map_beta_dust)
    hp.write_map(prefix_out+"/map_beta_dust_sigD%d_sd%d.fits"%(o.sigma_dust, o.seed), map_beta_dust, overwrite=True) 

if o.calculate_beta_sync:
    cl_betaSync = powerlaw(ells, gamma_beta_sync)
    delta_mapS = hp.synfast(cl_betaSync, nside, new=True, verbose=False)
    sigma_mapS = np.std(delta_mapS)
    delta_betaS = delta_mapS * ((o.sigma_sync/100) / sigma_mapS)
    #print(delta_betaS)
    mean_addS = -3
    map_beta_sync = delta_betaS+mean_addS
    #print(map_beta_sync)
    hp.write_map(prefix_out+"/map_beta_sync_sigS%d_sd%d.fits"%(o.sigma_sync, o.seed), map_beta_sync, overwrite=True) 





