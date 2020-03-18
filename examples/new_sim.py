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

if len(sys.argv) != 7:
    print("Usage: new_sim.py seed simulate nside pysm do_cl beta_var")
    exit(1)
seed = int(sys.argv[1])
do_simulation = int(sys.argv[2])
nside = int(sys.argv[3])
use_pysm = int(sys.argv[4])
do_Cls =  int(sys.argv[5])
beta_var =  int(sys.argv[6])
np.random.seed(seed)

prefix_in='/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/'
#'/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/template_PySM/'
prefix_out="./"

# Dust
A_dust_BB=5.0
EB_dust=2.  
alpha_dust_EE=-0.42 
alpha_dust_BB=-0.2
nu0_dust=353. 
temp_dust = 19.6
if beta_var:
    beta_dust =  hp.ud_grade(hp.read_map(prefix_in+'map_beta_dust2.fits', verbose=False), nside_out=nside)
else:
    beta_dust = 1.59

# Sync
A_sync_BB=2.0
EB_sync=2.
alpha_sync_EE=-0.6
alpha_sync_BB=-0.4
nu0_sync=23. 
if beta_var:
    beta_sync = hp.ud_grade(hp.read_map(prefix_in+'map_beta_sync2.fits', verbose=False), nside_out=nside)  
else:
    beta_sync=-3.

nu = np.array([27., 39., 93., 145., 225., 280.]) # elements [0,1,2,3,4,5]

def fcmb(nu):
    x=0.017608676067552197*nu
    ex=np.exp(x)
    return ex*(x/(ex-1))**2

if use_pysm:
    d2 = models("d2", nside) 
    s1 = models("s1", nside) 
    c1 = models("c1", nside)

    d2[0]['spectral_index'] = beta_dust
    d2[0]['temp'] = temp_dust * np.ones(d2[0]['temp'].size)
    s1[0]['spectral_index'] = beta_sync
    
    def power_law(nu, nu0, b):
        return (nu/nu0)**b

    def black_body(nu, nu0, t):
        return B(nu, t)/B(nu0, t)
    
    def B(nu, t):
        x = constants.h*nu*1E9 / constants.k / t
        return  (nu)**3  / np.expm1(x)

    def dust_sed(nu, nu0, beta, t):
        return power_law(nu, nu0, beta-2)*black_body(nu, nu0, t)

    if beta_var:
        dirname = prefix_out+"/new_sim_ns%d_seed%d_pysm_betaVar"%(nside, seed)
        for i in beta_dust:
            f_dust = dust_sed(nu, nu0_dust, i, temp_dust)
        for j in beta_sync:
            f_sync = power_law(nu, nu0_sync, j)
    else:
        dirname = prefix_out+"/new_sim_ns%d_seed%d_pysm"%(nside, seed)
        f_dust = dust_sed(nu, nu0_dust, beta_dust, temp_dust)
        f_sync = power_law(nu, nu0_sync, beta_sync)

else:
    #All spectra
    def comp_sed(nu,nu0,beta,temp,typ):
        if typ=='cmb':
            return fcmb(nu)
        elif typ=='dust':
            x_to=0.04799244662211351*nu/temp
            x_from=0.04799244662211351*nu0/temp
            return (nu/nu0)**(1+beta)*(np.exp(x_from)-1)/(np.exp(x_to)-1)
        elif typ=='sync':
            return (nu/nu0)**beta
        return None

    if beta_var:
        dirname = prefix_out+"/new_sim_ns%d_seed%d_bbsim_betaVar"%(nside, seed)
        for i in beta_dust:
            f_dust = comp_sed(nu, nu0_dust, i, temp_dust, 'dust')
        for j in beta_sync:
            f_sync = comp_sed(nu, nu0_sync, j, None, 'sync')
    else:
        dirname = prefix_out+"/new_sim_ns%d_seed%d_bbsim"%(nside, seed)
        f_dust = comp_sed(nu, nu0_dust, beta_dust, temp_dust, 'dust')
        f_sync = comp_sed(nu, nu0_sync, beta_sync, None, 'sync')

os.system('mkdir -p '+dirname)

f_cmb = fcmb(nu)

f_dust = f_dust/f_cmb
f_sync = f_sync/f_cmb
f_cmb = f_cmb/f_cmb

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
C_ells_sky = np.zeros([6, 2, 6, 2, len(ells)])
# EE
# [6,6,Nl]
C_ells_sky[:, 0, :, 0, :] = (cl_cmb_ee[None, None, :] * f_cmb[:, None, None] * f_cmb[None, :, None] +
                             cl_dust_ee[None, None, :] * f_dust[:, None, None] * f_dust[None, :, None] +
                             cl_sync_ee[None, None, :] * f_sync[:, None, None] * f_sync[None, :, None]) * cl2dl[None, None, :]
# BB
# [6,6,Nl]
C_ells_sky[:, 1, :, 1, :] = (cl_cmb_bb[None, None, :] * f_cmb[:, None, None] * f_cmb[None, :, None] +
                             cl_dust_bb[None, None, :] * f_dust[:, None, None] * f_dust[None, :, None] +
                             cl_sync_bb[None, None, :] * f_sync[:, None, None] * f_sync[None, :, None]) * cl2dl[None, None, :]

# Add noise
sens=1
knee=1
ylf=1
fsky=1.
nell=np.zeros([len(nu),lmax+1])
_,nell[:,2:],_=Simons_Observatory_V3_SA_noise(sens,knee,ylf,fsky,lmax+1,1)
nell*=cl2dl[None,:]

N_ells_sky = np.zeros([6, 2, 6, 2, len(ells)])
for i,n in enumerate(nu):
    for j in [0,1]:
        N_ells_sky[i, j, i, j, :] = nell[i]

Npix = hp.nside2npix(nside)
if do_simulation:
    maps_comp = np.zeros([3, 2, Npix])
    maps_comp[0,:,:] = hp.synfast([cl_cmb_tt, cl_cmb_ee, cl_cmb_bb, cl_cmb_te, cl_cmb_eb, cl_cmb_tb], nside, new=True)[1:]
    maps_comp[1,:,:] = hp.synfast([cl_dust_tt, cl_dust_ee, cl_dust_bb, cl_dust_te, cl_dust_eb, cl_dust_tb], nside, new=True)[1:]
    maps_comp[2,:,:] = hp.synfast([cl_sync_tt, cl_sync_ee, cl_sync_bb, cl_sync_te, cl_sync_eb, cl_sync_tb], nside, new=True)[1:]
    sed_comp = np.array([f_cmb, f_dust, f_sync]).T
    maps_signal = np.sum(maps_comp[None, :,:,:]*sed_comp[:,:,None,None], axis=1)

    maps_noise = np.zeros([6,2,Npix])
    for i in range(6):
        nell_ee = N_ells_sky[i, 0, i, 0, :]*dl2cl
        nell_bb = N_ells_sky[i, 1, i, 1, :]*dl2cl
        nell_00 = nell_ee * 0
        maps_noise[i, :, :] = hp.synfast([nell_00, nell_ee, nell_bb, nell_00, nell_00, nell_00], nside, new=True)[1:]

    if do_Cls:
        def map2cl(maps):
            cl_out = np.zeros([6,2,6,2,len(ells)])
            for i in range(6):
                m1 = np.zeros([3, Npix])
                m1[1:,:]=maps[i, :, :]
                for j in range(i,6):
                    print(i,j)
                    m2 = np.zeros([3, Npix])
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
        hp.write_map(dirname+"/maps_sky_signal.fits", maps_signal.reshape([len(nu)*2,Npix]) ,
                     overwrite=True)
        hp.write_map(dirname+"/maps_comp_cmb.fits", maps_comp[0],overwrite=True)
        hp.write_map(dirname+"/maps_comp_dust.fits", maps_comp[1], overwrite=True)
        hp.write_map(dirname+"/maps_comp_dust.fits", maps_comp[2], overwrite=True)
else:
    S_ells_sim = C_ells_sky
N_ells_sim = N_ells_sky
C_ells_sim = C_ells_sky + N_ells_sky

if do_Cls:
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
