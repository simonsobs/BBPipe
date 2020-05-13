import healpy as hp
import numpy as np
import scipy.constants as constants
import os
from noise_calc import Simons_Observatory_V3_SA_noise,Simons_Observatory_V3_SA_beams
from optparse import OptionParser
import pysm
from pysm.nominal import models

parser = OptionParser()
parser.add_option('--seed', dest='seed',  default=1200, type=int,
                  help='Set to define seed, default=1200')
parser.add_option('--nside', dest='nside', default=256, type=int,
                  help='Set to define Nside parameter, default=256')
parser.add_option('--pysm-sim', dest='pysm_sim', default=True, action='store_true',
                  help='Set to use PySM for simulations, default=True')
parser.add_option('--smooth', dest='use_smoothing', default=False, action='store_true',
                  help='Set to smooth beam, default=False')
parser.add_option('--beta-dust-gaus', dest='beta_dust_gaus', default=False, action='store_true',
                  help='Set to include gaussian spectral index for dust, default=False')
parser.add_option('--beta-sync-gaus', dest='beta_sync_gaus', default=False, action='store_true',
                  help='Set to include gaussian spectral index for sync, default=False')
parser.add_option('--beta-pysm', dest='beta_pysm', default=False,  action='store_true',
                  help='Set to include non-gaussian varying spectral indices, default=False')
parser.add_option('--sigma-d', dest='sigma_dust', default=0,  type=int,
                  help='Select amplitude of dust variation, default=0. Input values are multiplied to E-2')
parser.add_option('--sigma-s', dest='sigma_sync', default=0,  type=int,
                  help='Select amplitude of sync variation, default=0. Input values are multiplied to E-2')
parser
(o, args) = parser.parse_args()

nside = o.nside
seed= o.seed
np.random.seed(seed)

prefix_in='/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/'
prefix_out="/mnt/extraspace/susanna/tests_BBPipe_BBMoments/simulations_new_maps/"

# Dust
A_dust_BB=5
EB_dust=2.  
alpha_dust_EE=-0.42 
alpha_dust_BB=-0.42
nu0_dust=353. 
temp_dust = 19.6
beta_dust_const = 1.6 #1.59
if o.beta_pysm:
        beta_dust = pysm.read_map(prefix_in+'template_PySM/dust_beta.fits', nside, field=0, pixel_indices=None, mpi_comm=None)
elif o.beta_dust_gaus:
        beta_dust =  hp.ud_grade(hp.read_map(prefix_in+'map_beta_dust_sigD%d_sd%d.fits'%(o.sigma_dust, seed), verbose=False), nside_out=nside)
else:
        beta_dust = beta_dust_const

# Sync
A_sync_BB=2
EB_sync=2.
alpha_sync_EE=-0.6
alpha_sync_BB=-0.6 #-0.4
nu0_sync=23.
beta_sync_const = -3 #1.59
if o.beta_sync_gaus:
        beta_sync = hp.ud_grade(hp.read_map(prefix_in+'map_beta_sync_sigS%d_sd%d.fits'%(o.sigma_sync, o.seed), verbose=False), nside_out=nside)
elif o.beta_pysm:
        beta_sync = pysm.read_map(prefix_in+'template_PySM/synch_beta.fits', nside, field=0, pixel_indices=None, mpi_comm=None)
else:
        beta_sync = beta_sync_const

nu = np.array([27., 39., 93., 145., 225., 280.])
nfreq = len(nu)

# Define CMB spectrum
def fcmb(nu):
    x=0.017608676067552197*nu
    ex=np.exp(x)
    return ex*(x/(ex-1))**2

def plaw(nu, nu0, b):
        return (nu/nu0)**b

def power_law(nu, nu0, b_list):
        output = []
        for b in np.array(b_list):
                new_elem = plaw(nu, nu0, b)
                output.append(new_elem)
        return (np.array(output).reshape([6*npix]))

def black_body(nu, nu0, t):
        return B(nu, t)/B(nu0, t)
    
def B(nu, t):
        x = constants.h*nu*1E9 / constants.k / t
        return  (nu)**3  / np.expm1(x)

# All spectra
# For constant betas
def comp_sed_const(nu,nu0,beta,temp,typ):
        if typ=='cmb':
                return fcmb(nu)
        elif typ=='dust':
                return plaw(nu, nu0, beta-2)*black_body(nu, nu0, temp)
        elif typ=='sync':
                return plaw(nu, nu0, beta)
        return None

f_dust_const = comp_sed_const(nu, nu0_dust, beta_dust_const, temp_dust, 'dust')
f_sync_const = comp_sed_const(nu, nu0_sync, beta_sync_const, None, 'sync')
f_cmb_const = comp_sed_const(nu, None, None, None, 'cmb')

f_dust_const /= f_cmb_const
f_sync_const /= f_cmb_const
f_cmb_const /= f_cmb_const

if o.pysm_sim:
        if o.beta_pysm:
                dirname = prefix_out+"new_sim_ns%d_seed%d_pysm_betapysm"%(nside, seed)
        else:
                dirname = prefix_out+"new_sim_ns%d_seed%d_pysm_sigD%dsigS%d"%(nside, seed, o.sigma_dust, o.sigma_sync)
else:
        dirname = prefix_out+"new_sim_ns%d_seed%d_bbsim_sigD%dsigS%d"%(nside, seed, o.sigma_dust, o.sigma_sync)

        if o.beta_dust_gaus or o.beta_sync_gaus or o.beta_pysm:
                # For varying betas
                def comp_sed(nu,nu0,beta,temp,typ):
                        if typ=='cmb':
                                return (fcmb(nu)*np.ones([npix,6])).reshape([6*npix])
                        elif typ=='dust':
                                return power_law(nu, nu0, beta-2)*((black_body(nu, nu0, temp)*np.ones([npix,6])).reshape([6*npix]))
                        elif typ=='sync':
                                return power_law(nu, nu0, beta*np.ones([npix, 6]))
                        return None

                f_dust = comp_sed(nu, nu0_dust, beta_dust, temp_dust, 'dust')
                f_sync = comp_sed(nu, nu0_sync, beta_sync, None, 'sync')
                f_cmb = comp_sed(nu, None, None, None, 'cmb')
        else:
                # For constant betas
                f_dust = f_dust_const
                f_sync = f_sync_const
                f_cmb = f_cmb_const

        f_dust = f_dust/f_cmb
        f_sync = f_sync/f_cmb
        f_cmb = f_cmb/f_cmb

os.system('mkdir -p '+dirname)
#dirnameMoments = dirname+"MomTrue"
#os.system('mkdir -p '+dirnameMoments)

ncomp = 3 
f_cmb_RJ = fcmb(nu)

A_sync_BB = A_sync_BB * fcmb(nu0_sync)**2
A_dust_BB = A_dust_BB * fcmb(nu0_dust)**2

lmax = 3*nside-1
ells = np.arange(lmax+1)
nells = len(ells)
dl2cl=2*np.pi/(ells*(ells+1.)); dl2cl[0]=1
cl2dl=(ells*(ells+1.))/(2*np.pi)
npol = 2

# Dust
dl_dust_bb = A_dust_BB * ((ells+0.00000000000001) / 80.)**alpha_dust_BB 
dl_dust_ee = EB_dust * A_dust_BB * ((ells+0.00000000000001) / 80.)**alpha_dust_EE
cl_dust_bb = dl_dust_bb * dl2cl
cl_dust_ee = dl_dust_ee * dl2cl
cl_dust_tt = 0 * cl_dust_bb
cl_dust_tb = 0 * cl_dust_bb
cl_dust_eb = 0 * cl_dust_bb
cl_dust_te = 0 * cl_dust_bb

# Sync
dl_sync_bb = A_sync_BB * ((ells+0.00000000000001) / 80.)**alpha_sync_BB 
dl_sync_ee = EB_sync * A_sync_BB * ((ells+0.00000000000001) / 80.)**alpha_sync_EE
cl_sync_bb = dl_sync_bb * dl2cl
cl_sync_ee = dl_sync_ee * dl2cl
cl_sync_tt = 0 * cl_sync_bb
cl_sync_tb = 0 * cl_sync_bb
cl_sync_eb = 0 * cl_sync_bb
cl_sync_te = 0 * cl_sync_bb

# CMB
l,dtt,dee,dbb,dte=np.loadtxt("/mnt/zfsusers/susanna/camb_lens_nobb.dat",unpack=True)

l=l.astype(int)
msk=l<=lmax
l=l[msk]

dltt=np.zeros(len(ells)); dltt[l]=dtt[msk]
dlee=np.zeros(len(ells)); dlee[l]=dee[msk]
dlbb=np.zeros(len(ells)); dlbb[l]=dbb[msk]
dlte=np.zeros(len(ells)); dlte[l]=dte[msk]  
cl_cmb_bb=dlbb*dl2cl
#cl_cmb_ee=dlee*dl2cl
cl_cmb_ee = 0 * cl_cmb_bb #Temporarily set to zero to avoid b-mode purification
cl_cmb_tt = 0 * cl_cmb_bb
cl_cmb_tb = 0 * cl_cmb_bb
cl_cmb_eb = 0 * cl_cmb_bb
cl_cmb_te = 0 * cl_cmb_bb

# Noise
sens=1
knee=1
ylf=1
fsky=0.1
nell=np.zeros([nfreq,lmax+1])
_,nell[:,2:],_=Simons_Observatory_V3_SA_noise(sens,knee,ylf,fsky,lmax+1,1)
nell*=cl2dl[None,:]

N_ells_sky = np.zeros([nfreq, npol, nfreq, npol, nells])
for i,n in enumerate(nu):
    for j in [0,1]:
        N_ells_sky[i, j, i, j, :] = nell[i]

# Do simulation
npix = hp.nside2npix(nside)
maps_comp = np.zeros([ncomp, npol+1, npix])
maps_comp[0,:,:] = hp.synfast([cl_cmb_tt, cl_cmb_ee, cl_cmb_bb, cl_cmb_te, cl_cmb_eb, cl_cmb_tb], nside, new=True)
maps_comp[1,:,:] = hp.synfast([cl_dust_tt, cl_dust_ee, cl_dust_bb, cl_dust_te, cl_dust_eb, cl_dust_tb], nside, new=True)
maps_comp[2,:,:] = hp.synfast([cl_sync_tt, cl_sync_ee, cl_sync_bb, cl_sync_te, cl_sync_eb, cl_sync_tb], nside, new=True)

d2 = models("d2", nside) 
s1 = models("s1", nside) 
c1 = models("c1", nside) 
    
map_I_dust, map_Q_dust, map_U_dust = maps_comp[1,:,:]
map_I_sync, map_Q_sync, map_U_sync = maps_comp[2,:,:]
map_I_cmb,map_Q_cmb,map_U_cmb = maps_comp[0,:,:]

# Dust
d2[0]['A_I'] = map_I_dust
d2[0]['A_Q'] = map_Q_dust
d2[0]['A_U'] = map_U_dust
d2[0]['spectral_index'] = beta_dust
d2[0]['temp'] = temp_dust * np.ones(d2[0]['temp'].size) #need array, no const value for temp with PySM
# Sync
s1[0]['A_I'] = map_I_sync
s1[0]['A_Q'] = map_Q_sync
s1[0]['A_U'] = map_U_sync
s1[0]['spectral_index'] = beta_sync
# CMB
c1[0]['A_I'] = map_I_cmb
c1[0]['model'] = 'pre_computed' #different output maps at different seeds 
c1[0]['A_Q'] = map_Q_cmb
c1[0]['A_U'] = map_U_cmb

sky_config = {'dust' : d2, 'synchrotron' : s1, 'cmb' : c1}

sky = pysm.Sky(sky_config)

beams=Simons_Observatory_V3_SA_beams() 
instrument_config = {
    'nside' : nside,
    'frequencies' : nu, #Expected in GHz 
    'use_smoothing' : o.use_smoothing,
    'beams' : beams, #Expected in arcmin 
    'add_noise' : False,
    'use_bandpass' : False,
    'channel_names' : ['LF1', 'LF2', 'MF1', 'MF2', 'UHF1', 'UHF2'],
    'output_units' : 'uK_RJ',
    'output_directory' : dirname,
    'output_prefix' : '/test_deltabpass_',
}
    
sky = pysm.Sky(sky_config)
instrument = pysm.Instrument(instrument_config)
maps_signal, _ = instrument.observe(sky, write_outputs=False)
maps_signal = maps_signal[:,1:,:]
maps_signal = maps_signal/f_cmb_RJ[:,None,None]

# Save map
nmaps = nfreq * npol
hp.write_map(dirname+"/maps_sky_signal.fits", maps_signal.reshape([nmaps,npix]),
             overwrite=True)
hp.write_map(dirname+"/maps_comp_cmb.fits", maps_comp[0],overwrite=True)
hp.write_map(dirname+"/maps_comp_dust.fits", maps_comp[1], overwrite=True)
hp.write_map(dirname+"/maps_comp_dust.fits", maps_comp[2], overwrite=True)

# Mask
nhits=hp.ud_grade(hp.read_map("norm_nHits_SA_35FOV.fits",  verbose=False),nside_out=nside)
nhits/=np.amax(nhits) 
fsky_msk=np.mean(nhits) 
nhits_binary=np.zeros_like(nhits)
inv_sqrtnhits=np.zeros_like(nhits)
inv_sqrtnhits[nhits>1E-3]=1./np.sqrt(nhits[nhits>1E-3])
nhits_binary[nhits>1E-3]=1 

# Noise maps
nsplits = 4
maps_noise = np.zeros([nsplits, nfreq, npol, npix])
for s in range(nsplits):
    for i in range(nfreq):
        nell_ee = N_ells_sky[i, 0, i, 0, :]*dl2cl *nsplits
        nell_bb = N_ells_sky[i, 1, i, 1, :]*dl2cl *nsplits
        nell_00 = nell_ee * 0 *nsplits
        maps_noise[s, i, :, :] = hp.synfast([nell_00, nell_ee, nell_bb, nell_00, nell_00, nell_00], nside, pol=False, new=True)[1:] 
        # mask
maps_noise *= inv_sqrtnhits
        
# Save splits
for s in range(nsplits):
    hp.write_map(dirname+"/obs_split%dof%d.fits.gz" % (s+1, nsplits),
                 ((maps_signal[:,:,:]+maps_noise[s,:,:,:])*nhits_binary).reshape([nmaps,npix]),
                 overwrite=True)

# Write splits list
f=open(dirname+"/splits_list.txt","w")
Xout=""
for i in range(nsplits):
    Xout += dirname+'/obs_split%dof%d.fits.gz\n' % (i+1, nsplits)
f.write(Xout)
f.close()


