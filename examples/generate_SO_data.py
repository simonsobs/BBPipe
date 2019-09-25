import numpy as np
import matplotlib.pyplot as plt
from noise_calc import Simons_Observatory_V3_SA_noise

def get_output_params(do_phase=False, do_angle=False, do_sinuous=False):
    if do_sinuous:
        prefix_out = "SO_V3_Mock1_sinuous"
    else:
        prefix_out = "SO_V3_Mock1_phase%d_angle%d"%(int(do_phase),int(do_angle))
    if do_angle:
        angles = [1.,-1.,1.,-1.,1.,-1.]
    else:
        angles = [0.,0.,0.,0.,0.,0.]

    phase_suffix = ''
    if do_phase:
        phase_suffix = ''
    else:
        phase_suffix = '_0'

    phase_nu=['./data/phase_3layer_lf'+phase_suffix+'.txt',
              './data/phase_3layer_lf'+phase_suffix+'.txt',
              './data/phase_3layer_mf'+phase_suffix+'.txt',
              './data/phase_3layer_mf'+phase_suffix+'.txt',
              './data/phase_3layer_uhf'+phase_suffix+'.txt',
              './data/phase_3layer_uhf'+phase_suffix+'.txt']

    if do_sinuous:
        if do_phase:
            print("wait fuck we don't know how to do phase and sinuous yet plz fix")
            exit()
        phase_nu = ['./data/sinuous.txt', 
                    './data/sinuous.txt',
                    './data/sinuous.txt', 
                    './data/sinuous.txt', 
                    './data/sinuous.txt', 
                    './data/sinuous.txt']

    return prefix_out,phase_nu,angles

# Choose here whether to include the effects of
#  - A frequency-dependent polarization angle (do_phase=True)
#  - A non-zero constant polarization angle (do_angle=True)
prefix_out,phase_nu,angles=get_output_params(do_phase=False, do_angle=False, do_sinuous=False)


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
    def __init__(self,name,fname,angle=0,phase_nu=None):
        self.name=name
        self.nu,self.bnu=np.loadtxt(fname,unpack=True)
        self.dnu=np.zeros_like(self.nu)
        self.dnu[1:]=np.diff(self.nu)
        self.dnu[0]=self.dnu[1]
        if phase_nu is not None:
            from scipy.interpolate import interp1d
            n,p=np.loadtxt(phase_nu,unpack=True)
            p=np.radians(p)
            pf=interp1d(n,p,bounds_error=False,fill_value=0)
            phi=pf(self.nu)
            self.phase=np.cos(2*phi)+1j*np.sin(2*phi)
        # CMB units
        self.bnu/=np.sum(self.dnu*self.bnu*self.nu**2*fcmb(self.nu))
        self.norm = 1./np.sum(self.dnu*self.bnu*self.phase*self.nu**2*fcmb(self.nu))
        self.angle=angle
        print(self.angle)
        self.rot=np.array([[np.cos(2*self.angle),np.sin(2*self.angle)],
                           [-np.sin(2*self.angle),np.cos(2*self.angle)]])

    def convolve_sed(self,f):
        sed_b=np.sum(self.dnu*self.bnu*self.phase*self.nu**2*f(self.nu))*self.norm
        return np.array([[sed_b.real,sed_b.imag],[-sed_b.imag,sed_b.real]])

    def rotate(self,cl,transpose=False):
        if transpose:
            clrot=np.einsum('ijl,kj',cl,self.rot)
        else:
            clrot=np.einsum('ij,jkl',self.rot,cl)
        return clrot



#All frequencies and bandpasses
tracer_names=np.array(['SO_LF1','SO_LF2','SO_MF1','SO_MF2','SO_UHF1','SO_UHF2'])
fnames=['./data/LF/LF1.txt',
        './data/LF/LF2.txt',
        './data/MF/MF1.txt',
        './data/MF/MF2.txt',
        './data/UHF/UHF1.txt',
        './data/UHF/UHF2.txt']
bpss=[Bpass(n,f,angle=np.radians(a),phase_nu=p)
      for f,n,a,p in zip(fnames,tracer_names,angles,phase_nu)]

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
A_sync_BB = 10.
#A_sync_BB = 2.0
EB_sync = 2.
alpha_sync_EE = -0.6
alpha_sync_BB = -0.4
beta_sync = -3.2
#beta_sync = -3.
nu0_sync = 23.

#A_dust_BB = 5.0
A_dust_BB = 25.
EB_dust = 2.
#alpha_dust_EE = -0.42
alpha_dust_EE = -0.4
alpha_dust_BB = -0.2
beta_dust = 1.59
temp_dust = 19.6
nu0_dust = 353.

fg_EB_intrinsic_fraction = 0.05
Alens=1.

#Bandpowers
dell=10
nbands=100
lmax=2+nbands*dell
larr_all=np.arange(lmax+1)
lbands=np.linspace(2,lmax,nbands+1,dtype=int)
leff=0.5*(lbands[1:]+lbands[:-1])
windows=np.zeros([nbands,lmax+1])
for b,(l0,lf) in enumerate(zip(lbands[:-1],lbands[1:])):
    windows[b,l0:lf]=1
    windows[b,:]/=np.sum(windows[b])

#Component power spectra
def dl_plaw(A,alpha,ls):
    mask = ls > 0
    plaw = np.zeros(len(ls))
    plaw[mask] = A * (ls[mask]/80.)**alpha
    return plaw

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

#Convolve with windows
bpw_comp=np.sum(dls_comp[:,:,:,:,None,:]*windows[None,None,None,None,:,:],axis=5)

#Band-convolved SEDs
nfreqs=len(bpss)
seds=np.zeros([3,nfreqs,2,2]) #[ncomp,n_nu]
for ib,b in enumerate(bpss):
    seds[0,ib]=b.convolve_sed(lambda nu : comp_sed(nu,None,None,None,'cmb'))
    seds[1,ib]=b.convolve_sed(lambda nu : comp_sed(nu,nu0_sync,beta_sync,None,'sync'))
    seds[2,ib]=b.convolve_sed(lambda nu : comp_sed(nu,nu0_dust,beta_dust,temp_dust,'dust'))

def rotate_cells_mat(mat1, mat2, cls):
    if mat1 is not None:
        cls=np.einsum('ijk,lk',cls,mat1)
    if mat2 is not None:
        cls=np.einsum('jk,ikl',mat2,cls)
    return cls

#Compute multi-frequency spectra
#                       cfpp cfpp cpcpl
bpw_freq_sig=np.einsum('imnk,jopl,ikjlq',seds,seds,bpw_comp)
#Apply polarization angle
for f1 in range(nfreqs):
    for f2 in range(nfreqs):
        cl=bpw_freq_sig[f1,:,f2,:,:]
        clrot=bpss[f1].rotate(bpss[f2].rotate(cl,transpose=True))
        bpw_freq_sig[f1,:,f2,:,:]=clrot

#Add noise
sens=1
knee=1
ylf=1
fsky=0.1
cl2dl=larr_all*(larr_all+1)/(2*np.pi)
nell=np.zeros([nfreqs,lmax+1])
_,nell[:,2:],_=Simons_Observatory_V3_SA_noise(sens,knee,ylf,fsky,lmax+1,1)
nell*=cl2dl[None,:]
n_bpw=np.sum(nell[:,None,:]*windows[None,:,:],axis=2)
bpw_freq_noi=np.zeros_like(bpw_freq_sig)
for ib,n in enumerate(n_bpw):
    bpw_freq_noi[ib,0,ib,0,:]=n_bpw[ib,:]
    bpw_freq_noi[ib,1,ib,1,:]=n_bpw[ib,:]
bpw_freq_tot=bpw_freq_sig+bpw_freq_noi

bpw_freq_tot=bpw_freq_tot.reshape([nfreqs*2,nfreqs*2,nbands])
bpw_freq_sig=bpw_freq_sig.reshape([nfreqs*2,nfreqs*2,nbands])
bpw_freq_noi=bpw_freq_noi.reshape([nfreqs*2,nfreqs*2,nbands])

nmaps=2*nfreqs
ncross=(nmaps*(nmaps+1))//2
indices_tr=np.triu_indices(nmaps)

#Vectorize and compute covariance
bpw_tot=np.zeros([ncross,nbands])
bpw_sig=np.zeros([ncross,nbands])
bpw_noi=np.zeros([ncross,nbands])
cov_bpw=np.zeros([ncross,nbands,ncross,nbands])
factor_modecount=1./((2*leff+1)*dell*fsky)
for ii,(i1,i2) in enumerate(zip(indices_tr[0],indices_tr[1])):
    bpw_tot[ii,:]=bpw_freq_tot[i1,i2,:]
    bpw_sig[ii,:]=bpw_freq_sig[i1,i2,:]
    bpw_noi[ii,:]=bpw_freq_noi[i1,i2,:]
    for jj,(j1,j2) in enumerate(zip(indices_tr[0],indices_tr[1])):
        covar=(bpw_freq_tot[i1,j1,:]*bpw_freq_tot[i2,j2,:]+
               bpw_freq_tot[i1,j2,:]*bpw_freq_tot[i2,j1,:])*factor_modecount
        cov_bpw[ii,:,jj,:]=np.diag(covar)
bpw_tot=bpw_tot.flatten()
bpw_sig=bpw_sig.flatten()
bpw_noi=bpw_noi.flatten()
cov_bpw=cov_bpw.reshape([ncross*nbands,ncross*nbands])


#Write in SACC format
import sacc

#Tracers
def get_tracer_from_Bpass(b):
    return sacc.Tracer(b.name,"spin2",b.nu,b.bnu,'SO_SAT')
tracers=[get_tracer_from_Bpass(b) for b in bpss]

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
for ic,c in enumerate(corr_ordering):
    s1,s2=c
    tn1=s1[:-2]
    q1=s1[-1]
    t1=np.where(tracer_names==tn1)[0][0]
    tn2=s2[:-2]
    q2=s2[-1]
    t2=np.where(tracer_names==tn2)[0][0]
    typ=q1+q2
    for ib,w in enumerate(windows):
        lmean=leff[ib]
        win=sacc.Window(larr_all,w)
        ls_arr.append(lmean)
        w_arr.append(win)
    q1_arr+=nbands*[q1]
    q2_arr+=nbands*[q2]
    t1_arr+=nbands*[t1]
    t2_arr+=nbands*[t2]
    typ_arr+=nbands*[typ]
bins=sacc.Binning(typ_arr,ls_arr,t1_arr,q1_arr,t2_arr,q2_arr,windows=w_arr)

#Write
s_d=sacc.SACC(tracers,bins,mean=v_signal,precision=precis,
              meta={'data_name':'SO_V3_Mock_no_noise_data'})
s_f=sacc.SACC(tracers,bins,mean=v_signal,
              meta={'data_name':'SO_V3_Mock_no_noise_fiducial'})
s_n=sacc.SACC(tracers,bins,mean=v_noise,
              meta={'data_name':'SO_V3_Mock_no_noise_noise'})

s_d.saveToHDF(prefix_out+".sacc")
s_d.printInfo()
s_f.saveToHDF(prefix_out+"_fiducial.sacc")
s_f.printInfo()
s_n.saveToHDF(prefix_out+"_noise.sacc")
s_n.printInfo()
