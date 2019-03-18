from bbpipe import PipelineStage
from .types import FitsFile, TextFile
import numpy as np
import pylab as pl
import pymaster as nmt
import healpy as hp
from fgbuster.cosmology import _get_Cl_cmb 
from fgbuster.mixingmatrix import MixingMatrix

def binning_definition(nside, lmin=2, lmin=200, nlb=[], custom_bins=False):
    if custom_bins:
        ells=np.arange(3*nside,dtype='int32') #Array of multipoles
        weights=(1.0/nlb)*np.ones_like(ells) #Array of weights
        bpws=-1+np.zeros_like(ells) #Array of bandpower indices
        i=0;
        while 10*(i+1)+lmin<lmax :
            bpws[10*ilmin:10*(i+1)+lmin]=i
            i+=1
        b=nmt.NmtBin(nside,bpws=bpws,ells=ells,weights=weights)
    else:
        b=nmt.NmtBin(nside, nlb=int(1./self.config['fsky']))
    return b

class BBClEstimation(PipelineStage):
    """
    Stage that performs estimate the angular power spectra of:
        * each of the sky components, foregrounds and CMB, as well as the cross terms
        * of the corresponding covariance
    """

    name='BBClEstimation'
    inputs=[('binary_mask_cut',FitsFile),('post_compsep_maps',FitsFile), ('post_compsep_cov',FitsFile),
            ('A_maxL',TextFile),('noise_maps',FitsFile)]
    outputs=[('Cl_clean', FitsFile),('Cl_cov_clean', FitsFile)]

    def run(self):

        clean_map = hp.read_map(self.get_input('post_compsep_maps'),verbose=False, field=None, h=False)
        cov_map = hp.read_map(self.get_input('post_compsep_cov'),verbose=False, field=None, h=False)
        A_maxL = np.loadtxt(self.get_input('A_maxL'))
        noise_maps=hp.read_map(self.get_input('noise_maps'),verbose=False, field=None)

        nside_map = hp.get_nside(clean_map[0])
        print('nside_map = ', nside_map)
        
        w=nmt.NmtWorkspace()
        b = binning_definition(self.config['nside'], lmin=self.conf['lmin'], lmax=self.conf['lmax'],\
                                         nlb=self.conf['nlb'], custom_bins=self.conf['custom_bins']):

        print('building mask ... ')
        mask =  hp.read_map(self.get_input('binary_mask_cut'))
        mask_apo = nmt.mask_apodization(mask, self.config['aposize'], apotype=self.config['apotype'])

        print('building ell_eff ... ')
        ell_eff = b.get_effective_ells()

        #Read power spectrum and provide function to generate simulated skies
        cltt,clee,clbb,clte = hp.read_cl(self.config['Cls_fiducial'])[:,:4000]
        mp_t_sim,mp_q_sim,mp_u_sim=hp.synfast([cltt,clee,clbb,clte], nside=nside_map, new=True, verbose=False)

        def get_field(mp_q,mp_u) :
            #This creates a spin-2 field with both pure E and B.
            f2y=nmt.NmtField(mask_apo,[mp_q,mp_u],purify_e=False,purify_b=True)
            return f2y

        #We initialize two workspaces for the non-pure and pure fields:
        f2y0=get_field(mask*mp_q_sim,mask*mp_u_sim)
        w.compute_coupling_matrix(f2y0,f2y0,b)

        #This wraps up the two steps needed to compute the power spectrum
        #once the workspace has been initialized
        def compute_master(f_a,f_b,wsp) :
            cl_coupled=nmt.compute_coupled_cell(f_a,f_b)
            cl_decoupled=wsp.decouple_cell(cl_coupled)
            return cl_decoupled

        ### compute noise bias in the comp sep maps
        Cl_cov_clean_loc = []
        for f in range(len(self.config['frequencies'])):
            fn = get_field(mask*noise_maps[3*f+1], mask*noise_maps[3*f+2])
            Cl_cov_clean_loc.append(1.0/compute_master(fn,fn, w)[3] )

        AtNA = np.einsum('fi, fl, fj -> lij', A_maxL, np.array(Cl_cov_clean_loc), A_maxL)
        inv_AtNA = np.linalg.inv(AtNA)
        Cl_cov_clean = np.diagonal(inv_AtNA, axis1=-2,axis2=-1)
        Cl_cov_clean = np.vstack((ell_eff,Cl_cov_clean.swapaxes(0,1)))

        ### compute power spectra of the cleaned sky maps
        ncomp = int(len(clean_map)/2)
        Cl_clean = [ell_eff] 
        components = []
        print('n_comp = ', ncomp)
        ind=0
        for comp_i in range(ncomp):
            # for comp_j in range(ncomp)[comp_i:]:
            comp_j = comp_i
            print('comp_i = ', comp_i)
            print('comp_j = ', comp_j)

            components.append(str((comp_i,comp_j))) 

            fyp_i=get_field(mask*clean_map[2*comp_i], mask*clean_map[2*comp_i+1])
            fyp_j=get_field(mask*clean_map[2*comp_j], mask*clean_map[2*comp_j+1])

            # fyp_cov_i=get_field(mask*sqrt_cov_map[2*comp_i,2*comp_i], mask*sqrt_cov_map[2*comp_i+1,2*comp_i+1])
            # fyp_cov_j=get_field(mask*sqrt_cov_map[2*comp_j,2*comp_j], mask*sqrt_cov_map[2*comp_j+1,2*comp_j+1])

            Cl_clean.append(compute_master(fyp_i, fyp_j, w)[3])
            # Cl_cov_clean.append(compute_master(fyp_cov_i,fyp_cov_j, w)[3] )
            ind += 1
        print('ind = ', ind)
        print('shape(Cl_clean) = ', len(Cl_clean))
        print('shape(array(Cl_clean)) = ', (np.array(Cl_clean)).shape)
        print('all components = ', components)
        print('saving to disk ... ')
        hp.fitsfunc.write_cl(self.get_output('Cl_clean'), np.array(Cl_clean), overwrite=True)
        hp.fitsfunc.write_cl(self.get_output('Cl_cov_clean'), np.array(Cl_cov_clean), overwrite=True)

if __name__ == '__main__':
    results = PipelineStage.main()
    