import numpy as np
import matplotlib.pyplot as plt
from noise_calc import Simons_Observatory_V3_SA_noise,Simons_Observatory_V3_SA_beams
import sys
import pymaster as nmt
import healpy as hp
import os

if len(sys.argv)!=6:
    print("Usage: generate_SO_maps.py isim nside sens knee nsplits")
    exit(1)

isim=int(sys.argv[1])
nside=int(sys.argv[2])
sens=int(sys.argv[3])
knee=int(sys.argv[4])
nsplits=int(sys.argv[5])
npix= hp.nside2npix(nside)

prefix_out="SO_V3_ns%d_sens%d_knee%d_%dsplit_Mock%04d" % (nside, sens, knee, nsplits, isim)

#CMB spectrum
def fcmb(nu):
    x=0.017608676067552197*nu
    ex=np.exp(x)
    return ex*(x/(ex-1))**2

#All spectra
def comp_sed(nu,nu0,beta,temp,typ):
    if typ=='cmb':
        return fcmb(nu)
    elif typ=='dust':
        x_to=0.04799244662211351*nu/temp
        x_from=0.04799244662211351*nu0/temp
        return (nu/nu0)**(1+beta)*(np.exp(x_from)-1)/(np.exp(x_to)-1)*fcmb(nu0)
    elif typ=='sync':
        return (nu/nu0)**beta*fcmb(nu0)
    return None

#Bandpasses 
class Bpass(object):
    def __init__(self,name,fname):
        self.name=name
        self.nu,self.bnu=np.loadtxt(fname,unpack=True)
        self.dnu=np.zeros_like(self.nu)
        self.dnu[1:]=np.diff(self.nu)
        self.dnu[0]=self.dnu[1]
        # CMB units
        self.norm = 1./np.sum(self.dnu*self.bnu*self.nu**2*fcmb(self.nu))

    def convolve_sed(self,f):
        return np.sum(self.dnu*self.bnu*self.nu**2*f(self.nu))*self.norm


#All frequencies and bandpasses
tracer_names=np.array(['SO_LF1','SO_LF2','SO_MF1','SO_MF2','SO_UHF1','SO_UHF2'])
fnames=['./data/LF/LF1.txt',
        './data/LF/LF2.txt',
        './data/MF/MF1.txt',
        './data/MF/MF2.txt',
        './data/UHF/UHF1.txt',
        './data/UHF/UHF2.txt']
bpss=[Bpass(n,f)
      for f,n in zip(fnames,tracer_names)]

#Add E and B to tracer names to enumerate all possible maps
map_names=[]
for t in tracer_names:
    map_names.append(t+'_E')
    map_names.append(t+'_B')
#Upper-triangular ordering will be used
corr_ordering=[]
for i1,t1 in enumerate(map_names):
    for t2 in map_names[i1:]:
        corr_ordering.append([t1,t2])


#Foreground model
EB_sync=1.
EB_dust=2.
A_sync_BB=2.0
A_dust_BB=5.0
alpha_sync=-0.6
alpha_dust=-0.42
beta_sync=-3.
beta_dust=1.59
temp_dust=19.6
nu0_dust=353.
nu0_sync=23.
Alens=0.5

#Bandpowers
lmax=3*nside-1
larr_all=np.arange(lmax+1)
dlfac=2*np.pi/(larr_all*(larr_all+1.)); dlfac[0]=1

#Component power spectra
def dl_plaw(A,alpha,ls):
    dl = A*((ls+0.001)/80.)**alpha
    dl[:2]=0
    return dl

def read_camb(fname):
    l,dtt,dee,dbb,dte=np.loadtxt(fname,unpack=True)
    l=l.astype(int)
    msk=l<=lmax
    l=l[msk]
    dltt=np.zeros(len(larr_all)); dltt[l]=dtt[msk]
    dlee=np.zeros(len(larr_all)); dlee[l]=dee[msk]
    dlbb=np.zeros(len(larr_all)); dlbb[l]=dbb[msk]
    dlte=np.zeros(len(larr_all)); dlte[l]=dte[msk]
    return dltt,dlee,dlbb,dlte

# Foreground power spectra
cls_sync_ee=dl_plaw(A_sync_BB*EB_sync,alpha_sync,larr_all)*dlfac
cls_sync_bb=dl_plaw(A_sync_BB,alpha_sync,larr_all)*dlfac
cls_dust_ee=dl_plaw(A_dust_BB*EB_dust,alpha_dust,larr_all)*dlfac
cls_dust_bb=dl_plaw(A_dust_BB,alpha_dust,larr_all)*dlfac
_,dls_cmb_ee,dls_cmb_bb,_=read_camb("./data/camb_lens_nobb.dat")
cls_cmb_ee=dls_cmb_ee*dlfac
cls_cmb_bb=dls_cmb_bb*dlfac
cls_zero = np.zeros(3*nside)

#Band-convolved SEDs
nfreqs=len(bpss)
seds=np.zeros([nfreqs,3])
for ib,b in enumerate(bpss):
    seds[ib,0]=b.convolve_sed(lambda nu : comp_sed(nu,None,None,None,'cmb'))
    seds[ib,1]=b.convolve_sed(lambda nu : comp_sed(nu,nu0_sync,beta_sync,None,'sync'))
    seds[ib,2]=b.convolve_sed(lambda nu : comp_sed(nu,nu0_dust,beta_dust,temp_dust,'dust'))


nhits=hp.ud_grade(hp.read_map("norm_nHits_SA_35FOV.fits",verbose=False),nside_out=nside)
nhits/=np.amax(nhits)
nhits_binary=np.zeros_like(nhits)
inv_sqrtnhits=np.zeros_like(nhits)
inv_sqrtnhits[nhits>1E-3]=1./np.sqrt(nhits[nhits>1E-3])
nhits_binary[nhits>1E-3]=1
fsky=np.mean(nhits)
#Add noise
ylf=1
nell=np.zeros([nfreqs,lmax+1])
_,nell[:,2:],_=Simons_Observatory_V3_SA_noise(sens,knee,ylf,fsky,lmax+1,1,include_kludge=False)
beams=Simons_Observatory_V3_SA_beams()
for i,(n,b) in enumerate(zip(nell,beams)):
    sig = b * np.pi/180./60/2.355
    bl = np.exp(-sig**2*larr_all*(larr_all+1))
    n *= bl # Remove beam
    n[:2]=n[2] # Pad to ell=0

print("Generating noise maps")
noimaps = np.zeros([nsplits,nfreqs,2,npix])
for s in range(nsplits):
    for f in range(nfreqs):
        noimaps[s,f,0,:]=hp.synfast(nell[f] * nsplits, nside, pol=False, verbose=False) * inv_sqrtnhits
        noimaps[s,f,1,:]=hp.synfast(nell[f] * nsplits, nside, pol=False, verbose=False) * inv_sqrtnhits
noi_coadd = np.mean(noimaps, axis=0)

# Component amplitudes
print("Generating component maps")
maps_comp = np.zeros([3, 2, npix])
_, maps_comp[0, 0, :], maps_comp[0, 1, :] = hp.synfast([cls_zero, cls_cmb_ee, cls_cmb_bb,
                                                        cls_zero, cls_zero, cls_zero],
                                                       nside, pol=True, verbose=False)
_, maps_comp[1, 0, :], maps_comp[1, 1, :] = hp.synfast([cls_zero, cls_sync_ee, cls_sync_bb,
                                                        cls_zero, cls_zero, cls_zero],
                                                       nside, pol=True, verbose=False)
_, maps_comp[2, 0, :], maps_comp[2, 1, :] = hp.synfast([cls_zero, cls_dust_ee, cls_dust_bb,
                                                        cls_zero, cls_zero, cls_zero],
                                                       nside, pol=True, verbose=False)
print("Extrapolating in frequency")
# Components scaled over frequencies
maps_comp_freq = seds[:,:,None,None] * maps_comp[None,:,:,:]
print("Adding components")
# Sum all components
maps_freq = np.sum(maps_comp_freq, axis=1)

print("Beam convolution")
# Beam-convolution
for f,b in enumerate(beams):
    fwhm = b * np.pi/180./60.
    for i in [0,1]:
        maps_freq[f,i,:] = hp.smoothing(maps_freq[f,i,:], fwhm=fwhm, verbose=False)

print("Saving data")
os.system('mkdir -p ' + prefix_out)
hp.write_map(prefix_out+"/components_cmb.fits",maps_comp[0], overwrite=True)
hp.write_map(prefix_out+"/components_sync.fits",maps_comp[1], overwrite=True)
hp.write_map(prefix_out+"/components_dust.fits",maps_comp[1], overwrite=True)
hp.write_map(prefix_out+"/compfreq_cmb.fits",maps_comp_freq[:,0,:,:].reshape([2*nfreqs,npix]),
             overwrite=True)
hp.write_map(prefix_out+"/sky_signal.fits", maps_freq.reshape([nfreqs*2,npix]),
             overwrite=True)
hp.write_map(prefix_out+"/obs_coadd.fits", ((maps_freq+noi_coadd)*nhits_binary).reshape([nfreqs*2,npix]),
             overwrite=True)
for s in range(nsplits):
    hp.write_map(prefix_out+"/obs_split%dof%d.fits" % (s+1, nsplits),
                 ((maps_freq[:,:,:]+noimaps[s,:,:,:])*nhits_binary).reshape([nfreqs*2,npix]),
                 overwrite=True)

for f in range(nfreqs):
    np.savetxt(prefix_out + "/cls_noise_b%d.txt" % (f+1),np.transpose([larr_all,nell[f]]))
np.savetxt(prefix_out + "/cls_cmb.txt",np.transpose([larr_all, cls_cmb_ee, cls_cmb_bb, cls_zero]))
np.savetxt(prefix_out + "/cls_sync.txt",np.transpose([larr_all, cls_sync_ee, cls_sync_bb, cls_zero]))
np.savetxt(prefix_out + "/cls_dust.txt",np.transpose([larr_all, cls_dust_ee, cls_dust_bb, cls_zero]))
np.savetxt(prefix_out + "/seds.txt", np.transpose(seds))
