import numpy as np
import matplotlib.pyplot as plt
from noise_calc import Simons_Observatory_V3_SA_noise,Simons_Observatory_V3_SA_beams
import sys
import pymaster as nmt
import healpy as hp
import os
import sacc
from scipy.interpolate import interp1d

if len(sys.argv)!=7:
    print("Usage: generate_SO_maps.py isim nside sens knee nsplits mask_type")
    exit(1)

isim=int(sys.argv[1])
nside=int(sys.argv[2])
sens=int(sys.argv[3])
knee=int(sys.argv[4])
nsplits=int(sys.argv[5])
mask_type=sys.argv[6]
npix= hp.nside2npix(nside)

prefix_out="/mnt/extraspace/damonge/SO/BBPipe/"
prefix_out+="SO_V3_ns%d_sens%d_knee%d_%dsplit_Mask%s_Mock%04d" % (nside, sens, knee, nsplits, mask_type, isim)
print(prefix_out)

if mask_type=='analytic':
    # Sky fraction and apodization scale
    fsky=0.1
    aposcale=np.radians(10.)

    # Coords of patch centre
    vec_centre=np.array([1.,0.,0.])
    # Patch radius
    theta_range=np.arccos(1-2*fsky)+aposcale/2
    # Distance to patch centre
    theta=np.arccos(np.sum(np.array(hp.pix2vec(nside,np.arange(npix)))*
                           vec_centre[:,None],axis=0))
    # Normalized distance to edge
    x=np.sqrt((1-np.cos(theta-theta_range))/(1-np.cos(aposcale)))

    # 1 everywhere
    mask=np.ones(npix)
    nhits=np.ones(npix)
    # Apodized region
    mask[x<1]=(x-np.sin(2*np.pi*x)/(2*np.pi))[x<1]
    # Zero beyond range
    mask[theta>theta_range]=0
    nhits[theta>theta_range]=0
    hp.write_map("masks_analytic.fits",mask[None,:]*np.ones(6)[:,None],overwrite=True)
else:
    nhits=hp.ud_grade(hp.read_map("norm_nHits_SA_35FOV.fits",verbose=False),nside_out=nside)
    nhits/=np.amax(nhits)
    fsky=np.mean(nhits)

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
Alens=1.0

#Bandpowers
lmax=3*nside-1
larr_all=np.arange(lmax+1)
dlfac=2*np.pi/(larr_all*(larr_all+1.)); dlfac[0]=1

#Component power spectra
def dl_plaw(A,alpha,ls,correct_first=True):
    dl = A*((ls+0.001)/80.)**alpha
    if correct_first:
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
cls_cmb_bb=dls_cmb_bb*dlfac*Alens
cls_zero = np.zeros(3*nside)

#Band-convolved SEDs
nfreqs=len(bpss)
seds=np.zeros([nfreqs,3])
for ib,b in enumerate(bpss):
    seds[ib,0]=b.convolve_sed(lambda nu : comp_sed(nu,None,None,None,'cmb'))
    seds[ib,1]=b.convolve_sed(lambda nu : comp_sed(nu,nu0_sync,beta_sync,None,'sync'))
    seds[ib,2]=b.convolve_sed(lambda nu : comp_sed(nu,nu0_dust,beta_dust,temp_dust,'dust'))

ells_bpw=np.array([6.5,16.5,26.5,36.5,46.5,56.5,66.5,76.5,86.5,96.5,
                   106.5,116.5,126.5,136.5,146.5,156.5,166.5,176.5,186.5,196.5,
                   206.5,216.5,226.5,236.5,246.5,256.5,266.5,276.5,286.5,296.5,
                   306.5,316.5,326.5,336.5,346.5,356.5,366.5,376.5,386.5,396.5,
                   406.5,416.5,426.5,436.5,446.5,456.5,466.5,476.5,486.5,496.5,
                   506.5,516.5,526.5,536.5,546.5,556.5,566.5,576.5,586.5,596.5,
                   606.5,616.5,626.5,636.5,646.5,656.5,666.5,676.5,686.5,696.5,
                   706.5,716.5,726.5,736.5,746.5,756.5,766.5,776.5,786.5,796.5,
                   806.5,816.5,826.5,836.5,846.5,856.5,866.5,876.5,886.5,896.5,
                   906.5,916.5,926.5,936.5,946.5,956.5,966.5,976.5,986.5,996.5,
                   1006.5,1016.5,1026.5,1036.5,1046.5,1056.5,1066.5,1076.5,1086.5,1096.5,
                   1106.5,1116.5,1126.5,1136.5,1146.5,1156.5,1166.5,1176.5,1186.5,1196.5,
                   1206.5,1216.5,1226.5,1236.5,1246.5,1256.5,1266.5,1276.5,1286.5,1296.5,
                   1306.5,1316.5,1326.5,1336.5,1346.5,1356.5,1366.5,1376.5,1386.5,1396.5,
                   1406.5,1416.5,1426.5,1436.5,1446.5,1456.5,1466.5,1476.5,1486.5,1496.5,
                   1506.5,1516.5,1526.5])
#ells_bpw=np.array([6.5,16.5,26.5,36.5,46.5,56.5,66.5,76.5,86.5,
#                   96.5,106.5,116.5,126.5,136.5,146.5,156.5,
#                   166.5,176.5])
n_bpw=len(ells_bpw)
dlfac_bpw=2*np.pi/(ells_bpw*(ells_bpw+1.));
bpw_sync_ee=dl_plaw(A_sync_BB*EB_sync,alpha_sync,ells_bpw,correct_first=False)*dlfac_bpw
bpw_sync_bb=dl_plaw(A_sync_BB,alpha_sync,ells_bpw,correct_first=False)*dlfac_bpw
bpw_dust_ee=dl_plaw(A_dust_BB*EB_dust,alpha_dust,ells_bpw,correct_first=False)*dlfac_bpw
bpw_dust_bb=dl_plaw(A_dust_BB,alpha_dust,ells_bpw,correct_first=False)*dlfac_bpw
dls_cmb_ee_f=interp1d(larr_all,dls_cmb_ee)
dls_cmb_bb_f=interp1d(larr_all,dls_cmb_bb)
bpw_cmb_ee=dls_cmb_ee_f(ells_bpw)*dlfac_bpw
bpw_cmb_bb=dls_cmb_bb_f(ells_bpw)*dlfac_bpw

bpw_zero = np.zeros(n_bpw)
bpw_model=np.zeros([nfreqs,2,nfreqs,2,n_bpw])
for b1 in range(nfreqs):
    for b2 in range(nfreqs):
        bpw_model[b1,0,b2,0,:]+=bpw_cmb_ee*seds[b1,0]*seds[b2,0]
        bpw_model[b1,1,b2,1,:]+=bpw_cmb_bb*seds[b1,0]*seds[b2,0]
        bpw_model[b1,0,b2,0,:]+=bpw_sync_ee*seds[b1,1]*seds[b2,1]
        bpw_model[b1,1,b2,1,:]+=bpw_sync_bb*seds[b1,1]*seds[b2,1]
        bpw_model[b1,0,b2,0,:]+=bpw_dust_ee*seds[b1,2]*seds[b2,2]
        bpw_model[b1,1,b2,1,:]+=bpw_dust_bb*seds[b1,2]*seds[b2,2]
tracers=[]
for b in range(nfreqs):
    T=sacc.Tracer("band%d"%(b+1),'CMBP',
                  bpss[b].nu,bpss[b].bnu,exp_sample='SO_SAT')
    T.addColumns({'dnu':bpss[b].dnu})
    tracers.append(T)
typ, ell, t1, q1, t2, q2 = [], [], [], [], [], []
pol_names=['E','B']
for i1 in range(2*nfreqs):
    b1=i1//2
    p1=i1%2
    for i2 in range(i1,2*nfreqs):
        b2=i2//2
        p2=i2%2
        ty=pol_names[p1]+pol_names[p2]
        for il,ll in enumerate(ells_bpw):
            ell.append(ll)
            typ.append(ty)
            t1.append(b1)
            t2.append(b2)
            q1.append('C')
            q2.append('C')
bins=sacc.Binning(typ,ell,t1,q1,t2,q2)
bpw_model=bpw_model.reshape([2*nfreqs,2*nfreqs,n_bpw])[np.triu_indices(2*nfreqs)].flatten()
mean_model=sacc.MeanVec(bpw_model)
sacc_model=sacc.SACC(tracers,bins,mean=mean_model)
os.system('mkdir -p ' + prefix_out)
sacc_model.saveToHDF(prefix_out+"/cells_model.sacc")

nhits_binary=np.zeros_like(nhits)
inv_sqrtnhits=np.zeros_like(nhits)
inv_sqrtnhits[nhits>1E-3]=1./np.sqrt(nhits[nhits>1E-3])
nhits_binary[nhits>1E-3]=1
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
        noimaps[s,f,0,:]=hp.synfast(nell[f] * nsplits, nside, pol=False, verbose=False, new=True) * inv_sqrtnhits
        noimaps[s,f,1,:]=hp.synfast(nell[f] * nsplits, nside, pol=False, verbose=False, new=True) * inv_sqrtnhits
noi_coadd = np.mean(noimaps, axis=0)

# Component amplitudes
print("Generating component maps")
maps_comp = np.zeros([3, 2, npix])
_, maps_comp[0, 0, :], maps_comp[0, 1, :] = hp.synfast([cls_zero, cls_cmb_ee, cls_cmb_bb,
                                                        cls_zero, cls_zero, cls_zero],
                                                       nside, pol=True, verbose=False,new=True)
_, maps_comp[1, 0, :], maps_comp[1, 1, :] = hp.synfast([cls_zero, cls_sync_ee, cls_sync_bb,
                                                        cls_zero, cls_zero, cls_zero],
                                                       nside, pol=True, verbose=False,new=True)
_, maps_comp[2, 0, :], maps_comp[2, 1, :] = hp.synfast([cls_zero, cls_dust_ee, cls_dust_bb,
                                                        cls_zero, cls_zero, cls_zero],
                                                       nside, pol=True, verbose=False,new=True)

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
#hp.write_map(prefix_out+"/components_cmb.fits.gz",maps_comp[0], overwrite=True)
#hp.write_map(prefix_out+"/components_sync.fits.gz",maps_comp[1], overwrite=True)
#hp.write_map(prefix_out+"/components_dust.fits.gz",maps_comp[1], overwrite=True)
#hp.write_map(prefix_out+"/compfreq_cmb.fits.gz",maps_comp_freq[:,0,:,:].reshape([2*nfreqs,npix]),
#             overwrite=True)
#hp.write_map(prefix_out+"/sky_signal.fits.gz", maps_freq.reshape([nfreqs*2,npix]),
#             overwrite=True)
#hp.write_map(prefix_out+"/obs_coadd.fits.gz", ((maps_freq+noi_coadd)*nhits_binary).reshape([nfreqs*2,npix]),
#             overwrite=True)
for s in range(nsplits):
    hp.write_map(prefix_out+"/obs_split%dof%d.fits.gz" % (s+1, nsplits),
                 ((maps_freq[:,:,:]+noimaps[s,:,:,:])*nhits_binary).reshape([nfreqs*2,npix]),
                 overwrite=True)

for f in range(nfreqs):
    np.savetxt(prefix_out + "/cls_noise_b%d.txt" % (f+1),np.transpose([larr_all,nell[f]]))
np.savetxt(prefix_out + "/cls_cmb.txt",np.transpose([larr_all, cls_cmb_ee, cls_cmb_bb, cls_zero]))
np.savetxt(prefix_out + "/cls_sync.txt",np.transpose([larr_all, cls_sync_ee, cls_sync_bb, cls_zero]))
np.savetxt(prefix_out + "/cls_dust.txt",np.transpose([larr_all, cls_dust_ee, cls_dust_bb, cls_zero]))
np.savetxt(prefix_out + "/seds.txt", np.transpose(seds))
# Write splits list
f=open(prefix_out+"/splits_list.txt","w")
stout=""
for i in range(nsplits):
    stout += prefix_out+'/obs_split%dof%d.fits.gz\n' % (i+1, nsplits)
f.write(stout)
f.close()
