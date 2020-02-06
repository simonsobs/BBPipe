import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt

def get_deriv(mp) :
    ns=hp.npix2nside(len(mp))
    l=np.arange(3*ns)
    alpha1i=np.sqrt(l*(l+1.))
    alpha2i=np.sqrt((l-1.)*l*(l+1.)*(l+2.))
    mpd1=hp.alm2map(hp.almxfl(hp.map2alm(mp),alpha1i),nside=ns,verbose=False)
    mpd2=hp.alm2map(hp.almxfl(hp.map2alm(mp),alpha2i),nside=ns,verbose=False)
    return mpd1,mpd2
                            
ZER0=1E-3
APOSCALE=10.
nside=512
nh=hp.ud_grade(hp.read_map("norm_nHits_SA_35FOV.fits"),nside_out=nside)
nhg=hp.smoothing(nh,fwhm=np.pi/180,verbose=False)
nhg[nhg<0]=0
nh/=np.amax(nh)
nhg/=np.amax(nhg)
mpb=np.zeros_like(nh); mpb[nh>ZER0]=1
mpbg=np.zeros_like(nhg); mpbg[nhg>ZER0]=1
print("Apodize 1")
msk=nmt.mask_apodization(mpb,APOSCALE,apotype='C1')
print("Apodize 2")
mskg=nmt.mask_apodization(mpbg,APOSCALE,apotype='C1')
print("Deriv")
mskd1,mskd2=get_deriv(nh*msk)
mskd1g,mskd2g=get_deriv(nhg*mskg)
hp.write_map("mask_apo%.1lf.fits"%APOSCALE,mskg*nhg)

hp.write_map("masks_SAT.fits",(mskg*nhg)[None,:]*np.ones(6)[:,None],overwrite=True)

hp.mollview(msk)
hp.mollview(mskd1)
hp.mollview(mskd2)
hp.mollview(mskg)
hp.mollview(mskd1g)
hp.mollview(mskd2g)
plt.show()
