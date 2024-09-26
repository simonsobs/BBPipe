import healpy as hp
import numpy as np
# import mk_noise_map2 as mknm
# import matplotlib.pyplot as pl
import pylab as pl
import pymaster as nmt

binary_mask='/Users/josquin1/Documents/Dropbox/SKY_SIMULATIONS/SO_PySM/sky/mask_04000.fits'
# binary_mask='/Users/josquin1/Documents/Dropbox/SKY_SIMULATIONS/SO_PySM/sky/mask_02000.fits'
norm_hits_map='/Users/josquin1/Documents/Dropbox/CNRS-CR2/Simons_Observatory/norm_nHits_SA_35FOV_G.fits'
Cl_BB_lens='/Users/josquin1/Documents/Dropbox/CNRS-CR2/Simons_Observatory/SO_SAT_forecasting_tool/templates/Cls_Planck2018_lensed_scalar.fits'
Cl_BB_prim_r1='/Users/josquin1/Documents/Dropbox/CNRS-CR2/Simons_Observatory/SO_SAT_forecasting_tool/templates/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits'

nside = 256#512#256#128#512#128#512
nside_ = 1024#256#256#512#256#128#1024#1024#256#512#256#128#512#128#512
aposize = 10.0#10.0#8
aposize_ = 10.0#10.0#8
apotype = 'C1'
thrsd=1e-2

##############################

def binning_definition(nside, lmin=2, lmax=200, nlb=[], custom_bins=False):
    if custom_bins:
        ells=np.arange(3*nside,dtype='int32') #Array of multipoles
        weights=(1.0/nlb)*np.ones_like(ells) #Array of weights
        bpws=-1+np.zeros_like(ells) #Array of bandpower indices
        i=0;
        while (nlb+1)*(i+1)+lmin<lmax :
            bpws[(nlb+1)*i+lmin:(nlb+1)*(i+1)+lmin]=i
            i+=1 
        ##### adding a trash bin 2<=ell<=lmin
        # bpws[lmin:(nlb+1)*i+lmin] += 1
        # bpws[2:lmin] = 0
        # weights[2:lmin]= 1.0/(lmin-2-1)
        b=nmt.NmtBin(nside,bpws=bpws, ells=ells, weights=weights)
    else:
        b=nmt.NmtBin(nside, nlb=int(1./self.config['fsky']))
    return b

def get_mask(nh, nside_out=512, zer0=1e-6) :
    """
    Generates inverse-variance mask from Nhits map
    nside_out : output resolution
    """
    # zer0=1E-6
    # nh=get_nhits(nside_out=nside_out)
    nh/=np.amax(nh)
    msk=np.zeros(len(nh))
    not0=np.where(nh>zer0)[0]
    msk[not0]=nh[not0]
    return msk

"""
1- Read in an N_hits map
2- Smooth that out with a reasonable kernel (I think the Nhits we got initially had Nside=64~1-deg pixels, so I smoothed it with a 4-deg FWHM kernel or something like that)
3- Make a binary mask out of that (e.g. cut wherever the smoothed Nhits is larger than some value)
4- Apodize this binary mask
5- Multiply the nhits map and the apodized binary mask
"""

nhits = hp.read_map(norm_hits_map)
# hp.mollview(nhits)
# nhits = hp.ud_grade(nhits,nside_out=256)
# hp.smoothing(nhits, fwhm=10*np.pi/180.0, verbose=False)
# hp.mollview(nhits)
nhits = hp.ud_grade(nhits,nside_out=nside_)
nh = get_mask(nhits, nside_out=nside_, zer0=thrsd)
# hp.mollview(nh)
nh = hp.smoothing(nh, fwhm=10*np.pi/180.0, verbose=False)
# hp.mollview(nh)
# pl.plot(nh)
# pl.show()
# exit()


w=nmt.NmtWorkspace()
b = binning_definition( nside, lmin=30, lmax=2*nside, nlb=10, custom_bins=True)

print('building mask ... ')
mask =  hp.read_map( binary_mask )
mask = hp.ud_grade(mask, nside_out=nside_)

# hp.mollview(nh)
####### smoothing hits
print('smoothing')
nh = hp.smoothing(nh, fwhm=1*np.pi/180.0, verbose=False)
nh[nh<0]=0
# hp.mollview(nh)
####### make a binary mask nh_
print('binary mask building')
nh/=np.amax(nh)
nh_=np.zeros_like(nh); nh_[nh>thrsd]=1
# hp.mollview(nh_)
# pl.show()
# exit()
####### apodization of this binary mask
print('apodization')
mask_apo = nmt.mask_apodization(nh_, aposize, apotype=apotype)
# hp.mollview(mask_apo)
####### apodization of this binary mask
print('apo * hits')
mask_apo *= nh
# nh_ *= nh
# print('apodization')
mask_apo = nmt.mask_apodization(mask_apo, aposize_, apotype=apotype)
# hp.mollview(mask_apo)
# pl.show()
# exit()
#### downgrading 
# print('dowgrading apo')
mask_apo = hp.ud_grade(mask_apo, nside_out=nside)


fsky_eff = np.mean(mask_apo)
print('fsky_eff = ', fsky_eff)
# np.savetxt(self.get_output('fsky_eff'), [fsky_eff])

print('building ell_eff ... ')
ell_eff = b.get_effective_ells()

#Read power spectrum and provide function to generate simulated skies
print('read cl and map sim')
cltt,clee,clbb,clte = hp.read_cl(Cl_BB_lens)[:,:4000]
mp_t_sim,mp_q_sim,mp_u_sim=hp.synfast([cltt,clee,clbb,clte], nside=nside, new=True, verbose=False)

def get_field(mp_q,mp_u,purify_e=False,purify_b=True) :
    #This creates a spin-2 field with both pure E and B.
    f2y=nmt.NmtField(mask_apo,[mp_q,mp_u],purify_e=purify_e,purify_b=purify_b)
    # hp.mollview(mask_apo, title='mask apo w/ get_field :)')
    # pl.show()
    # exit()
    return f2y

#We initialize two workspaces for the non-pure and pure fields:
# if ((self.config['noise_option']!='white_noise') and (self.config['noise_option']!='no_noise')):
    # f2y0=get_field(mask_nh*mp_q_sim,mask_nh*mp_u_sim)
# else:
print('get field')
# f2y0=get_field(mask*mp_q_sim,mask*mp_u_sim)
f2y0=get_field(mp_q_sim,mp_u_sim)#,purify_b=False)

print('coupling matrix')
w.compute_coupling_matrix(f2y0,f2y0,b)

#This wraps up the two steps needed to compute the power spectrum
#once the workspace has been initialized
def compute_master(f_a,f_b,wsp) :
    cl_coupled=nmt.compute_coupled_cell(f_a,f_b)
    cl_decoupled=wsp.decouple_cell(cl_coupled)
    return cl_decoupled

##############################
# simulation of the CMB
print('simulations and plots')

Cl_BB_reconstructed = []
for i in range(10):
    print('simulation # ', i)
    mp_t_sim,mp_q_sim,mp_u_sim=hp.synfast([cltt,clee,clbb,clte], nside=nside, new=True, verbose=False)
    # f2y0=get_field(mask*mp_q_sim,mask*mp_u_sim, purify_b=True)
    f2y0=get_field(mp_q_sim,mp_u_sim)#, purify_b=False)
    Cl_BB_reconstructed.append(compute_master(f2y0, f2y0, w)[3])

np.save('Cl_BB_reconstructed', Cl_BB_reconstructed)
# Cl_BB_reconstructed = np.load('Cl_BB_reconstructed.npy')

pl.figure()
pl.loglog( ell_eff, np.array(Cl_BB_reconstructed).T, 'k-', alpha=0.2)
pl.loglog( ell_eff, np.mean(Cl_BB_reconstructed, axis=0), 'k--', alpha=1.0)
pl.loglog( ell_eff, b.bin_cell(clbb[:3*nside]), 'r--')
# pl.savefig('./test.pdf')
pl.show()
exit()
