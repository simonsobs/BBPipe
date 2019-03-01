from bbpipe import PipelineStage
from .types import FitsFile, TextFile
import numpy as np
import pylab as pl
import pymaster as nmt
import healpy as hp

class BBClEstimation(PipelineStage):
    """
    Stage that performs estimate the angular power spectra of:
        * each of the sky components, foregrounds and CMB, as well as the cross terms
        * of the corresponding covariance
    """

    name='BBClEstimation'
    inputs=[('binary_mask',FitsFile),('post_compsep_maps',FitsFile), ('post_compsep_cov',FitsFile)]
    outputs=[('Cl_clean', FitsFile),('Cl_cov_clean', FitsFile)]

    def run(self):

        clean_map = hp.read_map(self.get_input('post_compsep_maps'),verbose=False, field=None, h=True)
        cov_map = hp.read_map(self.get_input('post_compsep_cov'),verbose=False, field=None, h=True)

        nside_map = hp.nside2npix(hp.get_nside(clean_map[0]))
        w=nmt.NmtWorkspace()
        b=nmt.NmtBin(nside_map, nlb=10) ### nlb to be defined

        print(self.config)
        exit()


        print('building mask ... ')
        mask =  hp.read_map(self.get_input('binary_mask'))
        mask_apo = nmt.mask_apodization(mask, 0.2, apotype='C1') ### change apodization scale and type of apodization

        print('building ell_eff ... ')
        ell_eff = b.get_effective_ells()
        
        def compute_master(fl1,fl2) :
            cl=w.decouple_cell(nmt.compute_coupled_cell(fl1,fl2))
            cell=[]
            cell.append(ell_eff)
            for c in cl :
                cell.append(c)
            cell=np.array(cell)
            return cell

        ncomp = int(len(clean_map)/2)
        Cl_clean = [] 
        Cl_cov_clean = []
        components = []
        for (comp_i) in range(ncomp):
            for comp_j in range(ncomp)[comp_i:]:

                print('building f ... ')
                f=nmt.NmtField(mask_apo,[mask*Q_patch_map,mask*U_patch_map],purify_b=False)

                print('building w ... ')
                w.compute_coupling_matrix(f,f,b)

                print('building f ... ')
                f_clean_map = nmt.NmtField(mask,[mask*clean_map,mask*clean_map],purify_b=False)
                f_cov_map = nmt.NmtField(mask,[mask*cov_map,mask*cov_map],purify_b=False)

                print('computing Cl_NaMaster ... ')
                components.append(str((comp_i,comp_j))) 
                Cl_clean.append(compute_master(f_cov_map,f_cov_map))
                Cl_cov_clean.append(compute_master(f_cov_map,f_cov_map))

        print('saving to disk ... ')
        print(components)
        hp.fitsfunc.write_cl(self.get_output('Cl_clean'), Cl_clean, overwrite=False)
        hp.fitsfunc.write_cl(self.get_output('Cl_cov_clean'), Cl_cov_clean, overwrite=False)
