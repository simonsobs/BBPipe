from bbpipe import PipelineStage
from .types import FitsFile, TextFile, NumpyFile
import numpy as np
import pylab as pl
import pymaster as nmt
import healpy as hp
from fgbuster.cosmology import _get_Cl_cmb 
from fgbuster.mixingmatrix import MixingMatrix
import scipy.constants as constants
from astropy.cosmology import Planck15
from . import mk_noise_map2 as mknm
import scipy
from .Cl_estimation import binning_definition, get_field, compute_master


class BBClEstimation(PipelineStage):
    """
    Stage that performs estimate the angular power spectra of:
        * each of the sky components, foregrounds and CMB, as well as the cross terms
        * of the corresponding covariance
    """

    name='BBClEstimation'
    inputs=[('binary_mask_cut',FitsFile), ('A_maxL',NumpyFile), ('norm_hits_map', FitsFile),\
                ('fsky_eff',TextFile), ('mask_apo', FitsFile) ]
    outputs=[('Cl_noise_bias', FitsFile)]

    def run(self):

        # load the mixing matrix
        A_maxL = np.load(self.get_input('A_maxL'))

        # build the map-based operator
        for p in obs_pix:
            for s in range(2):
                noise_cov_inv = np.diag(1.0/noise_cov__[:,s,p])
                inv_AtNA = np.linalg.inv(A_maxL.T.dot(noise_cov_inv).dot(A_maxL))
                noise_after_comp_sep[:,s,p] = inv_AtNA.dot( A_maxL.T ).dot(noise_cov_inv).dot(noise_maps_[:,s,p])


        w=nmt.NmtWorkspace()

        
        # loop over simulations
        for i_sim in range(self.config['Nsims_bias']):
            
            # generating noise simulations
            nhits, noise_maps, nlev = mknm.get_noise_sim(sensitivity=self.config['sensitivity_mode'], 
                            knee_mode=self.config['knee_mode'],ny_lf=self.config['ny_lf'],
                                nside_out=self.config['nside'], norm_hits_map=hp.read_map(self.get_input('norm_hits_map')),
                                    no_inh=self.config['no_inh'])



            ### for comparison, compute the power spectrum of the noise after comp sep
            ### compute power spectra of the cleaned sky maps
            ncomp = int(len(clean_map)/2)
            Cl_clean = [ell_eff] 
            Cl_noise = [ell_eff] 
            components = []
            print('n_comp = ', ncomp)
            ind=0
            for comp_i in range(ncomp):
                # for comp_j in range(ncomp)[comp_i:]:
                comp_j = comp_i
                print('comp_i = ', comp_i)
                print('comp_j = ', comp_j)

                components.append(str((comp_i,comp_j))) 

                ## noise spectra
                fyp_i_noise=get_field(mask*post_compsep_noise[2*comp_i], mask*post_compsep_noise[2*comp_i+1], purify_b=True)
                fyp_j_noise=get_field(mask*post_compsep_noise[2*comp_j], mask*post_compsep_noise[2*comp_j+1], purify_b=True)

                Cl_noise.append(compute_master(fyp_i_noise, fyp_j_noise, w)[3])

                ind += 1

        hp.fitsfunc.write_cl(self.get_output('Cl_noise_bias'), np.array(Cl_noise_bias), overwrite=True)


if __name__ == '__main__':
    results = PipelineStage.main()
    