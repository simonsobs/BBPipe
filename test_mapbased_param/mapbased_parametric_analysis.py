from bbpipe import PipelineStage
from .types import FitsFile, TextFile
import numpy as np
import pylab as pl
import fgbuster as fg
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.separation_recipies import weighted_comp_sep
from fgbuster.algebra import Wd

class BBMapParamCompSep(PipelineStage):
    """
    Stage that performs three things:
        * fit of the spectral indices, using fgbuster tools, along with their error bars
        * estimate sky components, foregrounds and CMB
        * estimate components' covariance
    """
    name='BBMapParamCompSep'
    inputs= [('binary_mask_cut',FitsFile),('frequency_maps',FitsFile),('noise_cov',FitsFile),('noise_maps',FitsFile)]
    outputs=[('post_compsep_maps',FitsFile), ('post_compsep_cov',FitsFile), ('fitted_spectral_parameters',TextFile), ('A_maxL',TextFile),('post_compsep_noise',FitsFile)]

    def run(self) :
        #Read input mask
        import healpy as hp #We will want to be more general than assuming HEALPix
        binary_mask=hp.read_map(self.get_input('binary_mask_cut'),verbose=False)

        #Read frequency maps and noise covariance
        frequency_maps=hp.read_map(self.get_input('frequency_maps'),verbose=False, field=None)
        noise_cov=hp.read_map(self.get_input('noise_cov'),verbose=False, field=None)
        noise_maps=hp.read_map(self.get_input('noise_maps'),verbose=False, field=None)

        # reorganization of maps
        instrument = {'frequencies':np.array(self.config['frequencies'])}
        ind = 0
        frequency_maps_ = np.zeros((len(instrument['frequencies']), 3, frequency_maps.shape[-1]))
        noise_maps_ = np.zeros((len(instrument['frequencies']), 3, frequency_maps.shape[-1]))
        noise_cov_ = np.zeros((len(instrument['frequencies']), 3, frequency_maps.shape[-1]))
        for f in range(len(instrument['frequencies'])) : 
            for i in range(3): 
                frequency_maps_[f,i,:] =  frequency_maps[ind,:]*1.0
                noise_maps_[f,i,:] =  noise_maps[ind,:]*1.0
                noise_cov_[f,i,:] = noise_cov[ind,:]*1.0
                ind += 1
        
        # removing I from all maps
        frequency_maps_ = frequency_maps_[:,1:,:]
        noise_maps_ = noise_maps_[:,1:,:]
        noise_cov_ = noise_cov_[:,1:,:]

        # perform component separation
        # assuming inhomogeneous noise

        components = [CMB(), Dust(150., temp=20.0), Synchrotron(150.)]

        options={'disp':False, 'gtol': 1e-6, 'eps': 1e-4, 'maxiter': 100, 'ftol': 1e-6 } 
        tol=1e-18
        method='TNC'

        res = fg.separation_recipies.weighted_comp_sep(components, instrument,
                     data=frequency_maps_, cov=noise_cov_, nside=self.config['nside_patch'], 
                        options=options, tol=tol, method=method)

        # save results
        # fits for components maps
        # and text file for spectral parameters with error bars?
        # or maps for everything maybe....

        if res.s.shape[1] == 1:
            optI = 1
            optQU = 0
        elif res.s.shape[1] == 2:
            optI = 0
            optQU = 1
        else: 
            optI = 1
            optQU = 1

        A = MixingMatrix(*components)
        A_ev = A.evaluator(instrument['frequencies'])
        A_maxL = A_ev(res.x)
        np.savetxt(self.get_output('A_maxL'), A_maxL)

        A_maxL_loc = np.zeros((2*len(instrument['frequencies']), 6))
        print('shape(A_maxL) = ',np.shape(A_maxL))
        noise_cov_diag = np.zeros((2*len(instrument['frequencies']), 2*len(instrument['frequencies']), noise_cov.shape[1]))
        noise_maps__ = np.zeros((2*len(instrument['frequencies']), noise_cov.shape[1]))
        for f in range(len(instrument['frequencies'])):
            A_maxL_loc[2*f,:2] = A_maxL[f,0]
            A_maxL_loc[2*f,2:4] = A_maxL[f,1]
            A_maxL_loc[2*f,4:] = A_maxL[f,2]
            noise_cov_diag[2*f,2*f,:] = noise_cov_[f,0,:]*1.0
            noise_cov_diag[2*f+1,2*f+1,:] = noise_cov_[f,1,:]*1.0
            noise_maps__[2*f,:] = noise_maps_[f,0,:]*1.0
            noise_maps__[2*f+1,:] = noise_maps_[f,1,:]*1.0

        print('shape(A_maxL_loc) = ',np.shape(A_maxL_loc))
        print('shape(noise_cov_diag) = ',np.shape(noise_cov_diag))
        print('shape(noise_maps_) = ',np.shape(noise_maps__))
        # define masking
        mask = noise_maps__ == hp.UNSEEN
        mask = ~(np.any(mask, axis=tuple(range(noise_maps__.ndim-1))))

        noise_after_comp_sep = res.s*0.0
        obs_pix = np.where(mask!=0.0)[0]
        for p in obs_pix:
            inv_AtNA = np.linalg.inv(A_maxL_loc.T.dot(1.0/noise_cov_diag[:,:,p]).dot(A_maxL_loc))
            noise_after_comp_sep[:,p] = inv_AtNA.dot( A_maxL_loc.T ).dot(noise_cov_diag[:,:,p]).dot(noise_maps__)

        hp.write_map(self.get_output('post_compsep_noise'), noise_after_comp_sep, overwrite=True)

        column_names = []
        [ column_names.extend( (('I_'+str(ch))*optI,('Q_'+str(ch))*optQU,('U_'+str(ch))*optQU)) for ch in A.components]
        column_names = [x for x in column_names if x]
        maps_estimated=res.s[:,:,:].reshape((res.s.shape[0]*res.s.shape[1], res.s.shape[2]))
        hp.write_map(self.get_output('post_compsep_maps'), maps_estimated, overwrite=True, column_names=column_names)

        cov_estimated = res.invAtNA[:,:,:,:].diagonal().swapaxes(-1,0).swapaxes(-1,1)
        cov_estimated = cov_estimated.reshape((res.s.shape[0]*res.s.shape[1], res.s.shape[2]))
        hp.write_map(self.get_output('post_compsep_cov'), cov_estimated, overwrite=True, column_names=column_names)

        column_names = []
        [column_names.append(str(A.params[i])) for i in range(len(A.params))]
        all_combinations = []
        [all_combinations.append(str(A.params[i])+' x '+str(A.params[j])) for i, j in zip(list(np.triu_indices(len(A.params))[0]),list(np.triu_indices(len(A.params))[1]) )]
        [column_names.append(all_combinations[i]) for i in range(len(all_combinations))]
        np.savetxt(self.get_output('fitted_spectral_parameters'), np.hstack((res.x,  list(res.Sigma[np.triu_indices(len(A.params))]))), comments=column_names)



if __name__ == '__main__':
    results = PipelineStage.main()
