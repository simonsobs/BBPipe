import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from noise_calc import Simons_Observatory_V3_SA_noise,Simons_Observatory_V3_SA_beams
import sys
import os
import scipy.constants as constants
import fgbuster.component_model as fgc
import pysm
from pysm.nominal import models

import warnings
warnings.simplefilter("ignore")

from optparse import OptionParser

parser = OptionParser()

parser.add_option('--seed', dest='seed',  default=1300, type=int,
                  help='Set to define seed, default=1300')
parser.add_option('--nside', dest='nside', default=256, type=int,
                  help='Set to define Nside parameter, default=256')
parser.add_option('--simulate', dest='do_simulation', default=True, action='store_true',
                  help='Simulation step, default=True')
parser.add_option('--pysm-sim', dest='pysm_sim', default=True, action='store_true',
                  help='Set to use PySM for simulations, default=False')
parser.add_option('--do-cl', dest='do_Cls', default=True,  action='store_true', 
                  help='Calculate power spectra and covariance matrix, default=True')
parser.add_option('--beta-dust-var', dest='beta_dust_var', default=False,  action='store_true',
                  help='Set to include gaussian spectral indices, default=False')
parser.add_option('--beta-sync-var', dest='beta_sync_var', default=False,  action='store_true',
                  help='Set to include gaussian spectral indices, default=False')
parser.add_option('--beta-pysm', dest='beta_pysm', default=False,  action='store_true',
                  help='Set to include non-gaussian varying spectral indices, default=False')
parser.add_option('--sigma-d', dest='sigma_dust', default=0,  type=int,
                  help='Modify amplitude of dust variation, default=0. Input values are multiplied to E-2')
parser.add_option('--sigma-s', dest='sigma_sync', default=0,  type=int,
                  help='Modify amplitude of sync variation, default=0. Input values are multiplied to E-2')

(o, args) = parser.parse_args()

nside = o.nside
seed = o.seed
np.random.seed(seed)

if len(args) != 11:
        #parser.error
        print("Default settings: --seed --nside --simulate --pysm --do-cl --beta-var --beta-pysm --sigma-d --sigma-s")
        print("Check default settings:  <script> -h")

prefix_in='/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/'
prefix_out="./"

# Dust
A_dust_BB=5
EB_dust=2.  
alpha_dust_EE=-0.42 
alpha_dust_BB=-0.2
nu0_dust=353. 
temp_dust = 19.6
beta_dust_const = 1.59
if o.beta_pysm:
        beta_dust = pysm.read_map(prefix_in+'template_PySM/dust_beta.fits', nside, field=0, pixel_indices=None, mpi_comm=None)
elif o.beta_dust_var:
        beta_dust =  hp.ud_grade(hp.read_map(prefix_in+'map_beta_dust_sigD%d_sd%d.fits'%(o.sigma_dust, seed), verbose=False), nside_out=nside)
else:
        beta_dust = beta_dust_const
    
# Sync
A_sync_BB=2
EB_sync=2.
alpha_sync_EE=-0.6
alpha_sync_BB=-0.4
nu0_sync=23.
beta_sync_const=-3.
if o.beta_sync_var:
        beta_sync = hp.ud_grade(hp.read_map(prefix_in+'map_beta_sync_sigS%d_sd%d.fits'%(o.sigma_sync, o.seed), verbose=False), nside_out=nside)
elif o.beta_pysm:
        beta_sync = pysm.read_map(prefix_in+'template_PySM/synch_beta.fits', nside, field=0, pixel_indices=None, mpi_comm=None)
else:
        beta_sync = beta_sync_const

nu = np.array([27., 39., 93., 145., 225., 280.]) 
npix = hp.nside2npix(nside)

# CMB spectrum
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
        if o.beta_dust_var or o.beta_sync_var or o.beta_pysm:
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
                f_dust = f_dust_const
                f_sync = f_sync_const
                f_cmb = f_cmb_const

        f_dust = f_dust/f_cmb
        f_sync = f_sync/f_cmb
        f_cmb = f_cmb/f_cmb

os.system('mkdir -p '+dirname)
dirnameMoments = dirname+"MomTrue"
os.system('mkdir -p '+dirnameMoments)

f_cmb_RJ = fcmb(nu)

A_sync_BB = A_sync_BB * fcmb(nu0_sync)**2
A_dust_BB = A_dust_BB * fcmb(nu0_dust)**2

lmax = 3*nside-1
ells = np.arange(lmax+1)
dl2cl=2*np.pi/(ells*(ells+1.)); dl2cl[0]=1
cl2dl=(ells*(ells+1.))/(2*np.pi)

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
cl_cmb_ee=dlee*dl2cl
cl_cmb_tt = 0 * cl_cmb_bb
cl_cmb_tb = 0 * cl_cmb_bb
cl_cmb_eb = 0 * cl_cmb_bb
cl_cmb_te = 0 * cl_cmb_bb

# Raw sky power spectra
# Generate with some nominal spectra assuming constant spectral indices
C_ells_sky = np.zeros([6, 2, 6, 2, len(ells)])
# EE [nfreq, nfreq, nell]
C_ells_sky[:, 0, :, 0, :] = (cl_cmb_ee[None, None, :] * f_cmb_const[:, None, None] * f_cmb_const[None, :, None] +
                             cl_dust_ee[None, None, :] * f_dust_const[:, None, None] * f_dust_const[None, :, None] +
                             cl_sync_ee[None, None, :] * f_sync_const[:, None, None] * f_sync_const[None, :, None]) * cl2dl[None, None, :]
# BB [nfreq, nfreq, nell]
C_ells_sky[:, 1, :, 1, :] = (cl_cmb_bb[None, None, :] * f_cmb_const[:, None, None] * f_cmb_const[None, :, None] +
                             cl_dust_bb[None, None, :] * f_dust_const[:, None, None] * f_dust_const[None, :, None] +
                             cl_sync_bb[None, None, :] * f_sync_const[:, None, None] * f_sync_const[None, :, None]) * cl2dl[None, None, :]

# Add noise
sens=1
knee=1
ylf=1
fsky=0.1
nell=np.zeros([len(nu),lmax+1])
_,nell[:,2:],_=Simons_Observatory_V3_SA_noise(sens,knee,ylf,fsky,lmax+1,1)
nell*=cl2dl[None,:]

N_ells_sky = np.zeros([6, 2, 6, 2, len(ells)])
for i,n in enumerate(nu):
    for j in [0,1]:
        N_ells_sky[i, j, i, j, :] = nell[i]

if o.do_simulation:
    maps_comp = np.zeros([3, 2+1, npix])
    maps_comp[0,:,:] = hp.synfast([cl_cmb_tt, cl_cmb_ee, cl_cmb_bb, cl_cmb_te, cl_cmb_eb, cl_cmb_tb], nside, new=True)
    maps_comp[1,:,:] = hp.synfast([cl_dust_tt, cl_dust_ee, cl_dust_bb, cl_dust_te, cl_dust_eb, cl_dust_tb], nside, new=True)
    maps_comp[2,:,:] = hp.synfast([cl_sync_tt, cl_sync_ee, cl_sync_bb, cl_sync_te, cl_sync_eb, cl_sync_tb], nside, new=True)

    if o.pysm_sim:
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
                    'use_smoothing' : False,
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
    else:
            sed_comp = np.array([f_cmb, f_dust, f_sync]).T # [3, n_nu, n_pix].T = [n_pix, n_nu, 3]
            # Swap the axes 0 and 1
            sed_comp = np.transpose(np.array([f_cmb, f_dust, f_sync]), axes=[1, 0, 2]) # [n_nu, 3, n_pix]
            maps_signal = np.sum(maps_comp[None,:,1:,:]*sed_comp[:,:,None,:], axis=1)

    maps_noise = np.zeros([6,2,npix])
    for i in range(6):
        nell_ee = N_ells_sky[i, 0, i, 0, :]*dl2cl
        nell_bb = N_ells_sky[i, 1, i, 1, :]*dl2cl
        nell_00 = nell_ee * 0
        maps_noise[i, :, :] = hp.synfast([nell_00, nell_ee, nell_bb, nell_00, nell_00, nell_00], nside, new=True)[1:]

    if o.do_Cls:
        def map2cl(maps):
            cl_out = np.zeros([6,2,6,2,len(ells)])
            for i in range(6):
                m1 = np.zeros([3, npix])
                m1[1:,:]=maps[i, :, :]
                for j in range(i,6):
                    print(i,j)
                    m2 = np.zeros([3, npix])
                    m2[1:,:]=maps[j, :, :]

                    cl = hp.anafast(m1, m2, iter=0)
                    cl_out[i, 0, j, 0] = cl[1] * cl2dl
                    cl_out[i, 1, j, 1] = cl[2] * cl2dl
                    #cl_out[i, 0, j, 1] = cl[4] * cl2dl
                    #cl_out[i, 1, j, 0] = cl[4] * cl2dl
                    if j!=i:
                        cl_out[j, 0, i, 0] = cl[1] * cl2dl
                        cl_out[j, 1, i, 1] = cl[2] * cl2dl
                        #cl_out[j, 0, i, 1] = cl[4] * cl2dl
                        #cl_out[j, 1, i, 0] = cl[4] * cl2dl
            return cl_out
        S_ells_sim = map2cl(maps_signal+maps_noise)-N_ells_sky
    else:
        # Save maps: 2d array (nmaps, npix) 
        hp.write_map(dirname+"/maps_sky_signal.fits", maps_signal.reshape([len(nu)*2,npix]) ,
                     overwrite=True)
        hp.write_map(dirname+"/maps_comp_cmb.fits", maps_comp[0],overwrite=True)
        hp.write_map(dirname+"/maps_comp_dust.fits", maps_comp[1], overwrite=True)
        hp.write_map(dirname+"/maps_comp_dust.fits", maps_comp[2], overwrite=True)
else:
    S_ells_sim = C_ells_sky
N_ells_sim = N_ells_sky
C_ells_sim = C_ells_sky + N_ells_sky

if o.do_Cls:
    # Bandpowers
    delta_ell = 10
    N_bins = (len(ells)-2)//delta_ell
    W = np.zeros([N_bins, len(ells)])
    for i in range(N_bins):
        W[i, 2+i*delta_ell:2+(i+1)*delta_ell] = 1. / delta_ell

    # Bandpower averaging    
    C_s = np.dot(S_ells_sim, W.T)
    avg_bp = np.dot(ells, W.T)
    C_n = np.dot(N_ells_sim, W.T)
    C_t = np.dot(C_ells_sim, W.T)

    # Vectorize bandpowers
    C_s = C_s.reshape([6*2, 6*2, N_bins])
    ind = np.triu_indices(6*2)
    C_n = C_n.reshape([6*2, 6*2, N_bins])
    C_t = C_t.reshape([6*2, 6*2, N_bins])

    # Compute Covariance matrix
    ncross = len(ind[0])
    bpw_tot=np.zeros([ncross,N_bins])
    bpw_sig=np.zeros([ncross,N_bins])
    bpw_noi=np.zeros([ncross,N_bins])
    cov_bpw=np.zeros([ncross,N_bins,ncross,N_bins])
    factor_modecount=1./((2*avg_bp+1)*delta_ell*fsky)
    for ii,(i1,i2) in enumerate(zip(ind[0],ind[1])):
        bpw_tot[ii,:]=C_t[i1,i2,:]
        bpw_sig[ii,:]=C_s[i1,i2,:]
        bpw_noi[ii,:]=C_n[i1,i2,:]
        for jj,(j1,j2) in enumerate(zip(ind[0],ind[1])):
            covar=(C_t[i1,j1,:]*C_t[i2,j2,:]+
                   C_t[i1,j2,:]*C_t[i2,j1,:])*factor_modecount
            cov_bpw[ii,:,jj,:]=np.diag(covar)
    bpw_tot=bpw_tot.flatten()
    bpw_sig=bpw_sig.flatten()
    bpw_noi=bpw_noi.flatten()
    cov_bpw=cov_bpw.reshape([ncross*N_bins,ncross*N_bins])

    #Write in SACC format
    import sacc

    #Tracers
    tracers=[]
    for n in nu:
        nus= np.array([n-1 ,n ,n+1])
        bnus=np.array([0, 1, 0])
        name = '%d' %n
        tracers.append(sacc.Tracer(name, "spin2", nus, bnus, 'SO_SAT'))

    #Vectors
    v_signoi=sacc.MeanVec(bpw_tot)
    v_signal=sacc.MeanVec(bpw_sig)
    v_noise=sacc.MeanVec(bpw_noi)

    #Covariance
    precis=sacc.Precision(cov_bpw,is_covariance=True)

    #Ordering
    typ_arr=[]
    ls_arr=[]
    t1_arr=[]
    t2_arr=[]
    q1_arr=[]
    q2_arr=[]
    w_arr=[]
    pol_names = ['E', 'B']
    for ii,(i1,i2) in enumerate(zip(ind[0],ind[1])):
        i_nu1 = i1//2
        i_nu2 = i2//2
        i_p1 = i1%2
        i_p2 = i2%2
        p1 = pol_names[i_p1]
        p2 = pol_names[i_p2]
        t1_arr += N_bins * [i_nu1]
        t2_arr += N_bins * [i_nu2]
        q1_arr += N_bins * [p1]
        q2_arr += N_bins * [p2]
        typ_arr += N_bins*[p1+p2]
        for ib,w in enumerate(W):
            lmean=avg_bp[ib]
            win=sacc.Window(ells,w)
            ls_arr.append(lmean)
            w_arr.append(win)    
        bins=sacc.Binning(typ_arr,ls_arr,t1_arr,q1_arr,t2_arr,q2_arr,windows=w_arr)

    #Write
    s_d=sacc.SACC(tracers,bins,mean=v_signal,precision=precis,
                  meta={'data_name':'Mock_no_noise_data'})
    s_f=sacc.SACC(tracers,bins,mean=v_signal,
                  meta={'data_name':'Mock_no_noise_fiducial'})
    s_n=sacc.SACC(tracers,bins,mean=v_noise,
                  meta={'data_name':'Mock_no_noise_noise'})
    
    s_d.saveToHDF(dirname+"/dataCl.sacc")
    s_d.printInfo()
    s_n.saveToHDF(dirname+"/noiseCl.sacc")
    s_n.printInfo()
    s_f.saveToHDF(dirname+"/fiducialCl.sacc")
    s_f.printInfo()

print(dirname)
