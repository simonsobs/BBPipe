import numpy as np
import matplotlib.pyplot as plt
import noise_calc as nc
import sacc

#Foreground model
A_sync_BB = 2.0
EB_sync = 2.
alpha_sync_EE = -0.6
alpha_sync_BB = -0.4
beta_sync = -3.
nu0_sync = 23.

A_dust_BB = 5.0
EB_dust = 2.
alpha_dust_EE = -0.42
alpha_dust_BB = -0.2
beta_dust = 1.59
temp_dust = 19.6
nu0_dust = 353.

Alens = 1.0


#CMB spectrum
def fcmb(nu):
    x = 0.017608676067552197*nu
    ex = np.exp(x)
    return ex*(x/(ex-1))**2


#All spectra
def comp_sed(nu,nu0,beta,temp,typ):
    if typ == 'cmb':
        return fcmb(nu)
    elif typ == 'dust':
        x_to=0.04799244662211351*nu/temp
        x_from=0.04799244662211351*nu0/temp
        return (nu/nu0)**(1+beta)*(np.exp(x_from)-1)/(np.exp(x_to)-1)*fcmb(nu0)
    elif typ == 'sync':
        return (nu/nu0)**beta*fcmb(nu0)
    return None


#Component power spectra
def dl_plaw(A,alpha,ls):
    return A*((ls+0.001)/80.)**alpha


def read_camb(fname):
    l,dtt,dee,dbb,dte = np.loadtxt(fname,unpack=True)
    l = l.astype(int)
    msk = l <= lmax
    l = l[msk]
    dltt = np.zeros(len(larr_all))
    dltt[l] = dtt[msk]
    dlee = np.zeros(len(larr_all))
    dlee[l] = dee[msk]
    dlbb = np.zeros(len(larr_all))
    dlbb[l] = dbb[msk]
    dlte = np.zeros(len(larr_all))
    dlte[l] = dte[msk]
    return dltt,dlee,dlbb,dlte


#Bandpasses 
class Bpass(object):
    def __init__(self,name,fname):
        self.name = name
        self.nu,self.bnu = np.loadtxt(fname,unpack=True)
        self.dnu = np.zeros_like(self.nu)
        self.dnu[1:] = np.diff(self.nu)
        self.dnu[0] = self.dnu[1]
        # CMB units
        norm = np.sum(self.dnu*self.bnu*self.nu**2*fcmb(self.nu))
        self.bnu /= norm

    def convolve_sed(self,f):
        sed = np.sum(self.dnu*self.bnu*self.nu**2*f(self.nu))


band_names = ['LF1', 'LF2', 'MF1', 'MF2', 'UHF1', 'UHF2']

# Bandpasses
bpss = {n: Bpass(n,f'data/bandpasses/{n}.txt') for n in band_names}

# Bandpowers
dell = 10
nbands = 100
lmax = 2+nbands*dell
larr_all = np.arange(lmax+1)
lbands = np.linspace(2,lmax,nbands+1,dtype=int)
leff = 0.5*(lbands[1:]+lbands[:-1])
windows = np.zeros([nbands,lmax+1])
for b,(l0,lf) in enumerate(zip(lbands[:-1],lbands[1:])):
    windows[b,l0:lf] = 1
    windows[b,:] /= np.sum(windows[b])
s_wins = sacc.BandpowerWindow(larr_all, windows.T)

# Beams
beams = {band_names[i]: b for i, b in enumerate(nc.Simons_Observatory_V3_SA_beams(larr_all))}

print("Calculating power spectra")
# Component spectra
dls_sync_ee=dl_plaw(A_sync_BB*EB_sync,alpha_sync_EE,larr_all)
dls_sync_bb=dl_plaw(A_sync_BB,alpha_sync_BB,larr_all)
dls_dust_ee=dl_plaw(A_dust_BB*EB_dust,alpha_dust_EE,larr_all)
dls_dust_bb=dl_plaw(A_dust_BB,alpha_dust_BB,larr_all)
_,dls_cmb_ee,dls_cmb_bb,_=read_camb("./data/camb_lens_nobb.dat")
dls_comp=np.zeros([3,2,3,2,lmax+1]) #[ncomp,np,ncomp,np,nl]
dls_comp[0,0,0,0,:]=dls_cmb_ee
dls_comp[0,1,0,1,:]=Alens*dls_cmb_bb
dls_comp[1,0,1,0,:]=dls_sync_ee
dls_comp[1,1,1,1,:]=dls_sync_bb
dls_comp[2,0,2,0,:]=dls_dust_ee
dls_comp[2,1,2,1,:]=dls_dust_bb

# Convolve with windows
bpw_comp=np.sum(dls_comp[:,:,:,:,None,:]*windows[None,None,None,None,:,:],axis=5)

# Convolve with bandpasses
nfreqs = len(band_names)
seds = np.zeros([3,nfreqs])
for ib, n in enumerate(band_names):
    b = bpss[n]
    seds[0,ib] = b.convolve_sed(lambda nu : comp_sed(nu,None,None,None,'cmb'))
    seds[1,ib] = b.convolve_sed(lambda nu : comp_sed(nu,nu0_sync,beta_sync,None,'sync'))
    seds[2,ib] = b.convolve_sed(lambda nu : comp_sed(nu,nu0_dust,beta_dust,temp_dust,'dust'))

# Component -> frequencies
bpw_freq_sig=np.einsum('ik,jm,iljno',seds,seds,bpw_comp)

# N_ell
sens=1
knee=1
ylf=1
fsky=0.1
cl2dl=larr_all*(larr_all+1)/(2*np.pi)
nell=np.zeros([nfreqs,lmax+1])
_,nell[:,2:],_=nc.Simons_Observatory_V3_SA_noise(sens,knee,ylf,fsky,lmax+1,1)
nell*=cl2dl[None,:]
n_bpw=np.sum(nell[:,None,:]*windows[None,:,:],axis=2)
bpw_freq_noi=np.zeros_like(bpw_freq_sig)
for ib,n in enumerate(n_bpw):
    bpw_freq_noi[ib,0,ib,0,:]=n_bpw[ib,:]
    bpw_freq_noi[ib,1,ib,1,:]=n_bpw[ib,:]

# Add to signal
bpw_freq_tot=bpw_freq_sig+bpw_freq_noi
bpw_freq_tot=bpw_freq_tot.reshape([nfreqs*2,nfreqs*2,nbands])
bpw_freq_sig=bpw_freq_sig.reshape([nfreqs*2,nfreqs*2,nbands])
bpw_freq_noi=bpw_freq_noi.reshape([nfreqs*2,nfreqs*2,nbands])


# Creating Sacc files
s_d = sacc.Sacc()
s_f = sacc.Sacc()
s_n = sacc.Sacc()

# Adding tracers
print("Adding tracers")
for n in band_names:
    bandpass = bpss[n]
    beam = beams[n]
    for s in [s_d, s_f, s_n]:
        s.add_tracer('NuMap', 'SAT_' + n,
                     quantity='cmb_polarization',
                     spin=2,
                     nu=bandpass.nu,
                     bandpass=bandpass.bnu,
                     ell=larr_all,
                     beam=beam,
                     nu_unit='GHz',
                     map_unit='uK_CMB')

# Adding power spectra
print("Adding spectra")
nmaps=2*nfreqs
ncross=(nmaps*(nmaps+1))//2
indices_tr=np.triu_indices(nmaps)
map_names=[]
for n in band_names:
    map_names.append('SAT_' + n + '_E')
    map_names.append('SAT_' + n + '_B')
for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
    n1 = map_names[i1][:-2]
    n2 = map_names[i2][:-2]
    p1 = map_names[i1][-1].lower()
    p2 = map_names[i2][-1].lower()
    cl_type = f'cl_{p1}{p2}'
    s_d.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_sig[i1, i2, :], window=s_wins)
    s_f.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_sig[i1, i2, :], window=s_wins)
    s_n.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_noi[i1, i2, :], window=s_wins)

# Add covariance
print("Adding covariance")
cov_bpw = np.zeros([ncross, nbands, ncross, nbands])
factor_modecount = 1./((2*leff+1)*dell*fsky)
for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
    for jj, (j1, j2) in enumerate(zip(indices_tr[0], indices_tr[1])):
        covar = (bpw_freq_tot[i1, j1, :]*bpw_freq_tot[i2, j2, :]+
                 bpw_freq_tot[i1, j2, :]*bpw_freq_tot[i2, j1, :]) * factor_modecount
        cov_bpw[ii, :, jj, :] = np.diag(covar)
cov_bpw = cov_bpw.reshape([ncross * nbands, ncross * nbands])
s_d.add_covariance(cov_bpw)

# Write output
print("Writing")
s_d.save_fits("cls_coadd.fits", overwrite=True)
s_f.save_fits("cls_fid.fits", overwrite=True)
s_n.save_fits("cls_noise.fits", overwrite=True)
