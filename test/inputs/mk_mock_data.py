import numpy as np
import healpy as hp
import V3calc as v3
import spectra as spc

ZER0=1E-3

def read_cl_teb(fname,lmax=3000) :
    """
    Read TEB power spectrum in CAMB format into l,C_l
    """
    data=np.loadtxt(fname,unpack=True)
    larr=data[0]
    clteb=np.zeros([len(larr)+2,4])
    clteb[2:,0]=data[1]*2*np.pi/(larr*(larr+1))
    clteb[2:,1]=data[2]*2*np.pi/(larr*(larr+1))
    clteb[2:,2]=data[3]*2*np.pi/(larr*(larr+1))
    clteb[2:,3]=data[4]*2*np.pi/(larr*(larr+1))
    
    return np.arange(lmax),np.transpose(clteb[:lmax])

def get_sky_sim_gaus(nside,
                     r_cmb,alens_cmb,
                     ABB_sync=3.8,AEE_sync=3.8,alpha_sync=-0.60,
                     beta_sync=-3.0,curv_sync= 0.,nu0_sync= 23.,
                     ABB_dust=3.8,AEE_dust=3.8,alpha_dust=-0.42,
                     beta_dust=1.59,temp_dust=20.,nu0_dust=353.,
                     freqs=[0.],seed_cmb=None,seed_fg=None,need_fg=True) :
    if seed_cmb is not None :
        np.random.seed(seed_cmb)
        
    nell=3*nside
    nnu=len(freqs)
    lp,cteb_prim=read_cl_teb("data/planck1_r1p00_tensCls.dat")
    ll,cteb_lens=read_cl_teb("data/planck1_r0p00_lensedtotCls.dat")
    clTT_cmb=(r_cmb*cteb_prim+          cteb_lens)[0,:nell]
    clEE_cmb=(r_cmb*cteb_prim+          cteb_lens)[1,:nell]
    clBB_cmb=(r_cmb*cteb_prim+alens_cmb*cteb_lens)[2,:nell]
    clTE_cmb=(r_cmb*cteb_prim+          cteb_lens)[3,:nell]
    spec_cmb=spc.cmb(freqs)
    clt_cmb=np.zeros([3,nnu,nnu,nell])
    clt_cmb[0]=clTT_cmb[None,None,:]*spec_cmb[:,None,None]*spec_cmb[None,:,None]
    clt_cmb[1]=clEE_cmb[None,None,:]*spec_cmb[:,None,None]*spec_cmb[None,:,None]
    clt_cmb[2]=clBB_cmb[None,None,:]*spec_cmb[:,None,None]*spec_cmb[None,:,None]
    input_cmb={'spec':spec_cmb,'ctt':clTT_cmb,'cte':clTE_cmb,'cee':clEE_cmb,'cbb':clBB_cmb,
               'ctot':clt_cmb,'cbbp':cteb_prim[2,:nell],'cbbl':cteb_lens[2,:nell]}

    if need_fg :
        if seed_fg is not None :
            np.random.seed(seed_fg)
        
        ell=np.arange(nell)
        dl_prefac=2*np.pi/((ell+0.01)*(ell+1))
        clZERO=np.zeros_like(ell)
        clTT_sync=clZERO
        clTE_sync=clZERO
        clBB_sync=dl_prefac*ABB_sync*((ell+0.1)/80.)**alpha_sync*spc.cmb(nu0_sync)**2
        clEE_sync=dl_prefac*AEE_sync*((ell+0.1)/80.)**alpha_sync*spc.cmb(nu0_sync)**2
        spec_sync=spc.sync_curvedpl(freqs,nu0_sync,beta_sync,curv_sync)
        clt_sync=np.zeros([3,nnu,nnu,nell])
        clt_sync[0]=clTT_sync[None,None,:]*spec_sync[:,None,None]*spec_sync[None,:,None]
        clt_sync[1]=clEE_sync[None,None,:]*spec_sync[:,None,None]*spec_sync[None,:,None]
        clt_sync[2]=clBB_sync[None,None,:]*spec_sync[:,None,None]*spec_sync[None,:,None]
        input_sync={'spec':spec_sync,'ctt':clTT_sync,'cte':clTE_sync,'cee':clEE_sync,'cbb':clBB_sync,
                    'ctot':clt_sync}

        clTT_dust=clZERO
        clTE_dust=clZERO
        clBB_dust=dl_prefac*ABB_dust*((ell+0.1)/80.)**alpha_dust*spc.cmb(nu0_dust)**2
        clEE_dust=dl_prefac*AEE_dust*((ell+0.1)/80.)**alpha_dust*spc.cmb(nu0_dust)**2
        spec_dust=spc.dustmbb(freqs,nu0_dust,beta_dust,temp_dust)
        clt_dust=np.zeros([3,nnu,nnu,nell])
        clt_dust[0]=clTT_dust[None,None,:]*spec_dust[:,None,None]*spec_dust[None,:,None]
        clt_dust[1]=clEE_dust[None,None,:]*spec_dust[:,None,None]*spec_dust[None,:,None]
        clt_dust[2]=clBB_dust[None,None,:]*spec_dust[:,None,None]*spec_dust[None,:,None]
        input_dust={'spec':spec_dust,'ctt':clTT_dust,'cte':clTE_dust,'cee':clEE_dust,'cbb':clBB_dust,
                    'ctot':clt_dust}

        amp_sync=np.array(hp.synfast([clTT_sync,clEE_sync,clBB_sync,clTE_sync,clZERO,clZERO],
                                     nside,pol=True,new=True,verbose=False))
        mp_sync=amp_sync[None,:,:]*spec_sync[:,None,None]
        amp_dust=np.array(hp.synfast([clTT_dust,clEE_dust,clBB_dust,clTE_dust,clZERO,clZERO],
                                     nside,pol=True,new=True,verbose=False))
        mp_dust=amp_dust[None,:,:]*spec_dust[:,None,None]
    else :
        input_sync=None; amp_sync=0; mp_sync=0; input_dust=None; amp_dust=0; mp_dust=0;
    amp_cmb=np.array(hp.synfast([clTT_cmb,clEE_cmb,clBB_cmb,clTE_cmb,clZERO,clZERO],
                                nside,pol=True,new=True,verbose=False))
    mp_cmb=amp_cmb[None,:,:]*spec_cmb[:,None,None]

    return mp_cmb,mp_dust+mp_sync

def get_nhits() :
    fname_out='data/norm_nHits_SA_35FOV_G.fits'
    return hp.ud_grade(hp.read_map(fname_out,verbose=False),
                       nside_out=nside)
def get_mask() :
    nh=get_nhits()
    nh/=np.amax(nh)
    msk=np.zeros(len(nh))
    not0=np.where(nh>ZER0)[0]
    msk[not0]=nh[not0]
    return msk

def get_noise_sim(nside,sensitivity=2,knee=1,ny_lf=1.,seed=None) :
    if seed is not None :
        np.random.seed(seed)
    freqs=v3.so_V3_SA_bands()
    nell=3*nside
    nnu=len(freqs)
    nh=get_nhits()
    msk=get_mask()
    fsky=np.mean(msk)
    ll,nll,nlev=v3.so_V3_SA_noise(sensitivity,knee,ny_lf,fsky,nell,remove_kluge=True)
    units=spc.cmb(v3.so_V3_SA_bands())

    cl_th=np.zeros([3,nnu,nnu,nell])
    for i_n,n in enumerate(nll) :
        nl=np.zeros(nell)
        nl[2:]=n; nl[:2]=n[0]
        cl_th[0,i_n,i_n,:]=nl/2.*units[i_n]**2
        cl_th[1,i_n,i_n,:]=nl*units[i_n]**2
        cl_th[2,i_n,i_n,:]=nl*units[i_n]**2
    
    id_cut=np.where(nh<ZER0)[0]
    nh[id_cut]=np.amax(nh)
    mps_no=[];
    for i_n in np.arange(len(nll)) :
        nl=cl_th[:,i_n,i_n,:]
        no_t,no_q,no_u=hp.synfast([nl[0],nl[1],nl[2],0*nl[0],0*nl[0],0*nl[0]],nside=nside,
                                  pol=True,new=True,verbose=False)
        no_t/=np.sqrt(nh/np.amax(nh));
        no_q/=np.sqrt(nh/np.amax(nh));
        no_u/=np.sqrt(nh/np.amax(nh));
        mps_no.append([no_t,no_q,no_u])
    mps_no=np.array(mps_no)

    input_noi={'ctot':cl_th}
        
    return mps_no,msk

nsplits=2
nside=64
nus=v3.so_V3_SA_bands()
mp_cmb,mp_fgs=get_sky_sim_gaus(nside,0,0.5,freqs=nus)

#Write maps
for isplit in range(nsplits) :
    mp_noi,mask=get_noise_sim(nside)
    for inu,nu in enumerate(nus) :
        hp.write_map("map_split%dof%d_nu%dof%d.fits"%(isplit+1,nsplits,inu+1,len(nus)),
                     mp_cmb[inu]+mp_fgs[inu]+mp_noi[inu],overwrite=True)

#Write mask
hp.write_map("mask.fits",mask,overwrite=True)
