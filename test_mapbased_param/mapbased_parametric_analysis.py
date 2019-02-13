from bbpipe import PipelineStage
from .types import FitsFile
import numpy as np
import pylab as pl


class BBMapParamCompSep(PipelineStage):
    """
    Stage that performs three things:
        * fit of the spectral indices, using fgbuster tools, along with their error bars
        * estimate sky components, foregrounds and CMB
        * estimate components' covariance
    """
    name='BBMapParamCompSep'
    inputs= [('binary_mask',FitsFile),('frequency_maps',FitsFile),('noise_cov',FitsFile)]
    outputs=[('post_compsep_maps',FitsFile)]

    def run(self) :
        #Read input mask
        import healpy as hp #We will want to be more general than assuming HEALPix
        binary_mask=hp.read_map(self.get_input('binary_mask'),verbose=False)

        #Read frequency maps and noise covariance
        frequency_maps=hp.read_map(self.get_input('frequency_maps'),verbose=False, field=None)
        noise_cov=hp.read_map(self.get_input('noise_cov'),verbose=False, field=None)

        # reorganization of maps
        instrument = {'frequencies':[30.0, 40.0, 95.0, 150.0, 220.0, 270.0]}
        ind = 0
        frequency_maps_ = np.zeros((len(instrument['frequencies']), 3, frequency_maps.shape[-1]))
        noise_cov_ = np.zeros((len(instrument['frequencies']), 3, frequency_maps.shape[-1]))
        for f in range(len(instrument['frequencies'])) : 
            for i in range(3): 
                frequency_maps_[f,i,:] =  frequency_maps[ind,:]*1.0
                noise_cov_[f,i,:] = noise_cov[ind,:]*1.0
                ind += 1
        # removing I from all maps
        frequency_maps_ = frequency_maps_[:,1:,:]
        noise_cov_ = noise_cov_[:,1:,:]

        # perform component separation
        # assuming inhomogeneous noise
        import fgbuster as fg
        from fgbuster.component_model import CMB, Dust, Synchrotron
        components = [CMB(), Dust(150., temp=20.0), Synchrotron(150.)]

        from fgbuster.mixingmatrix import MixingMatrix
        A = MixingMatrix(*components)
        A_ev = A.evaluator(instrument['frequencies'])
        print(A_ev(np.array([1.54, -3.0])))
        A_dB_ev = A.diff_evaluator(instrument['frequencies'])
        print(A_dB_ev(np.array([1.54, -3.0])))

        from fgbuster.separation_recipies import weighted_comp_sep
        res = fg.separation_recipies.weighted_comp_sep(components, instrument,
                     data=frequency_maps_, cov=noise_cov_, nside=0)

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
        
        from fgbuster.mixingmatrix import MixingMatrix
        A = MixingMatrix(*components)
        column_names = []
        [ column_names.extend( ('I_'+str(ch)*optI,'Q_'+str(ch)+'GHz','U_'+str(ch)+'GHz')) for ch in A.components]
        
        maps_estimated=res.s[:,:,:].reshape((res.s.shape[0]*res.s.shape[1], res.s.shape[2]))
        hp.write_map(self.get_output('post_compsep_maps'), maps_estimated, overwrite=True, column_names=column_names)

        cov_estimated = res.invAtNA[:,:,:,:].diagonal()
        cov_estimated= cov_estimated.reshape((res.s.shape[0]*res.s.shape[1], res.s.shape[2]))
        hp.write_map(self.get_output('post_compsep_cov'), cov_estimated, overwrite=True, column_names=column_names)

        column_names = []
        [column_names.append(str(A.params[i])) for i in range(len(A.params))]
        all_combinations = []
        [all_combinations.append(str(A.params[i])+' x '+str(A.params[j])) for i, j in zip(list(np.triu_indices(len(A.params))[0]),list(np.triu_indices(len(A.params))[1]) )]
        [column_names.append(all_combinations[i]) for i in range(len(all_combinations))]
        np.savetxt(self.get_output('fitted_spectral_parameters'), np.hstack((res.x,  list(res.Sigma[np.triu_indices(len(A.params))]))), comments=column_names)

if __name__ == '__main__':
    results = PipelineStage.main()
