import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import V3calc as v3
import os

def get_nhits(nside_out=512) :
    """
    Generates an Nhits map in Galactic coordinates.
    nside_out : output resolution
    """
    fname_out='norm_nHits_SA_35FOV_G_nside512.fits'

    if not os.path.isfile(fname_out) :
        fname_in='norm_nHits_SA_35FOV.fits'
        mp_C=hp.read_map(fname_in,verbose=False)
        nside_l=hp.npix2nside(len(mp_C))

        nside_h=512
        ipixG=np.arange(hp.nside2npix(nside_h))
        thG,phiG=hp.pix2ang(nside_h,ipixG)
        r=hp.Rotator(coord=['G','C'])
        thC,phiC=r(thG,phiG)
        ipixC=hp.ang2pix(nside_l,thC,phiC)

        mp_G=hp.ud_grade(mp_C[ipixC],nside_out=nside_l)
        hp.write_map(fname_out,mp_G)

    return hp.ud_grade(hp.read_map(fname_out,verbose=False),
                       nside_out=nside_out)

def get_mask(nh, nside_out=512) :
    """
    Generates inverse-variance mask from Nhits map
    nside_out : output resolution
    """
    zer0=1E-6
    # nh=get_nhits(nside_out=nside_out)
    nh/=np.amax(nh)
    msk=np.zeros(len(nh))
    not0=np.where(nh>zer0)[0]
    msk[not0]=nh[not0]
    return msk

def get_noise_sim(sensitivity=2,knee_mode=1,ny_lf=1.,nside_out=512, norm_hits_map=None, no_inh=False) :
    """
    Generates noise simulation
    sensitivity : choice of sensitivity model for SAC's V3 
    knee_mode : choice of ell_knee model for SAC's V3
    ny_lf : number of years with an LF tube
    nside_out : output resolution
    """
    print('norm_hits_map = ', norm_hits_map)
    if norm_hits_map is None:
        nh=get_nhits(nside_out=nside_out)
    else:
        nh = hp.ud_grade(norm_hits_map,nside_out=nside_out)

    msk=get_mask(nh, nside_out=nside_out)
    fsky=np.mean(msk)

    print(sensitivity,knee_mode,ny_lf,fsky,3*nside_out)

    ll,nll,nlev=v3.so_V3_SA_noise(sensitivity,knee_mode,ny_lf,fsky,3*nside_out,remove_kluge=True)
    zer0=1E-6
    id_cut=np.where(nh<zer0)[0]
    nh[id_cut]=np.amax(nh) #zer0
    mps_no=[];
    for i_n in np.arange(len(nll)) :
        n=nll[i_n]
        nl=np.zeros(3*nside_out)
        nl[2:]=n; nl[:2]=n[0]
        no_t,no_q,no_u=hp.synfast([nl/2.,nl,nl,0*nl,0*nl,0*nl],nside=nside_out,
                                  pol=True,new=True,verbose=False)
        # nv_t=nlev[i_n]*np.ones_like(no_t)/np.sqrt(2.);
        # nv_q=nlev[i_n]*np.ones_like(no_q); nv_u=nlev[i_n]*np.ones_like(no_u)
        if not no_inh: 
            no_t/=np.sqrt(nh/np.amax(nh))
            no_q/=np.sqrt(nh/np.amax(nh))
            no_u/=np.sqrt(nh/np.amax(nh));
        # nv_t/=np.sqrt(nh/np.amax(nh)); nv_q/=np.sqrt(nh/np.amax(nh)); nv_u/=np.sqrt(nh/np.amax(nh));
        # mps_no.append([no_t,no_q,no_u])
        mps_no.append(no_t)
        mps_no.append(no_q)
        mps_no.append(no_u)

    mps_no=np.array(mps_no)
    return msk,mps_no,nlev

# nside_run=512
# mask,maps,nlev=get_noise_sim()

# mp_plot=maps[3][1]*mask
# mp_plot[mask<=0]=hp.UNSEEN

# hp.mollview(mp_plot,title='Noise (* inv. var. mask) @ 150 GHz, Q',coord=['G','C'])
# plt.savefig('noise.png',bbox_inches='tight')
# plt.show()
