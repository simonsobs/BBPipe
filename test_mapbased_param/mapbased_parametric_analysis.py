from bbpipe import PipelineStage
from .types import FitsFile, TextFile, NumpyFile
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
    inputs= [('binary_mask_cut',FitsFile),('frequency_maps',FitsFile),('noise_cov',FitsFile),
                ('noise_maps',FitsFile)]
    outputs=[('post_compsep_maps',FitsFile), ('post_compsep_cov',FitsFile), ('fitted_spectral_parameters',TextFile),
                 ('A_maxL',NumpyFile),('post_compsep_noise',FitsFile), ('mask_patches', FitsFile)]

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

        if self.config['Nspec']!=0.0:
            print('Nspec != 0 !! building and analysis independent Bd regions ... ')
            # read the template used to produce simulation
            # in practice, we would use the Bd map estimated from data sets.
            Bd_template = hp.read_map('/global/cscratch1/sd/josquin/SO_sims/dust_beta.fits')
            # upgrade the map to the actual working resolution
            Bd_template = hp.ud_grade(Bd_template, nside_out=self.config['nside'])
            # make slices through this map. Define the regions of interest
            mask_patches = np.zeros((self.config['Nspec'], len(Bd_template)))
            # observed patches
            obs_pix = np.where(binary_mask!=0.0)[0]
            # thickness of the corresponding patches
            delta_Bd_patch = np.abs(np.max(Bd_template[obs_pix])-np.min(Bd_template[obs_pix]))/(self.config['Nspec'])
            # definition of slices so that it includes the max of Bd as the last step
            slices = np.arange(np.min(Bd_template[obs_pix]), np.max(Bd_template[obs_pix])+delta_Bd_patch/10.0, delta_Bd_patch )
            for i in range(self.config['Nspec']):
                pix_within_patch = np.where((Bd_template[obs_pix] >= slices[i] ) & (Bd_template[obs_pix] < slices[i+1]))[0]
                mask_patches[i,obs_pix[pix_within_patch]] = 1
        else:
            mask_patches = binary_mask[np.newaxis,:]

        noise_after_comp_sep_ = np.zeros((6, noise_cov.shape[1]))
        maps_estimated = np.zeros((6, noise_cov.shape[1]))
        cov_estimated = np.zeros((6, 6, noise_cov.shape[1]))

        # loop over pixels within defined-above regions:
        resx = []
        resS = []

        for i_patch in range(mask_patches.shape[0]):

            mask_patch_ = mask_patches[i_patch]

            # filtering masked regions of the patch ... 
            frequency_maps__ = frequency_maps_*1.0
            frequency_maps__[:,:,np.where(mask_patch_==0)[0]] = hp.UNSEEN
            noise_cov__ = noise_cov_*1.0
            noise_cov__[:,:,np.where(mask_patch_==0)[0]] = hp.UNSEEN

            res = fg.separation_recipies.weighted_comp_sep(components, instrument,
                         data=frequency_maps__, cov=noise_cov__, nside=self.config['nside_patch'], 
                            options=options, tol=tol, method=method)

            resx.append(res.x)
            resS.append(res.Sigma)

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

            if i_patch == 0 :
                A_maxL_v = np.zeros((mask_patches.shape[0], A_maxL.shape[0], A_maxL.shape[1]))
            A_maxL_v[i_patch,:,:] = A_maxL*1.0

            A_maxL_loc = np.zeros((2*len(instrument['frequencies']), 6))

            # reshaping quantities of interest
            noise_cov_diag = np.zeros((2*len(instrument['frequencies']), 2*len(instrument['frequencies']), noise_cov.shape[1]))
            noise_maps__ = np.zeros((2*len(instrument['frequencies']), noise_cov.shape[1]))

            for f in range(len(instrument['frequencies'])):
                A_maxL_loc[2*f,:2] = A_maxL[f,0]
                A_maxL_loc[2*f+1,:2] = A_maxL[f,0]
                A_maxL_loc[2*f,2:4] = A_maxL[f,1]
                A_maxL_loc[2*f+1,2:4] = A_maxL[f,1]
                A_maxL_loc[2*f,4:] = A_maxL[f,2]
                A_maxL_loc[2*f+1,4:] = A_maxL[f,2]
                noise_cov_diag[2*f,2*f,:] = noise_cov__[f,0,:]*1.0
                noise_cov_diag[2*f+1,2*f+1,:] = noise_cov__[f,1,:]*1.0
                noise_maps__[2*f,:] = noise_maps_[f,0,:]*1.0
                noise_maps__[2*f+1,:] = noise_maps_[f,1,:]*1.0

            # define masking
            # mask_patch_ = (noise_maps__[0] == hp.UNSEEN )#| noise_maps__[0] == 0.0)
            noise_after_comp_sep = np.zeros((3,2, noise_cov.shape[1]))#*hp.UNSEEN
            obs_pix = np.where(mask_patch_==1.0)[0]

            for p in obs_pix:
                for s in range(2):
                    noise_cov_inv = np.diag(1.0/noise_cov__[:,s,p])
                    inv_AtNA = np.linalg.inv(A_maxL.T.dot(noise_cov_inv).dot(A_maxL))
                    noise_after_comp_sep[:,s,p] = inv_AtNA.dot( A_maxL.T ).dot(noise_cov_inv).dot(noise_maps_[:,s,p])

            # the noise if the combination (sum) of the noise_after_comp_sep
            # recovered on each of the sub masks
            for f in range(noise_after_comp_sep.shape[0]):
                noise_after_comp_sep_[2*f,:] += noise_after_comp_sep[f,0,:]*1.0
                noise_after_comp_sep_[2*f+1,:] += noise_after_comp_sep[f,1,:]*1.0

            # reshape map_estimated_ from the recovered sky signals ... 
            # set to zeros areas with hp.UNSEEN
            ress = res.s[:,:,:]
            for i in range(ress.shape[0]):
                for j in range(ress.shape[1]):
                    ress[i,j,np.where(ress[i,j,:]==hp.UNSEEN)[0]] = 0.0
            maps_estimated += ress.reshape((res.s.shape[0]*res.s.shape[1], res.s.shape[2]))

            # reshaping and saving the covariance matrix
            '''
            cov_estimated_ = res.invAtNA[:,:,:,:].diagonal().swapaxes(-1,0).swapaxes(-1,1)
            cov_estimated_reshaped = cov_estimated_.reshape((res.s.shape[0]*res.s.shape[1], res.s.shape[2]))
            for i in range(cov_estimated_reshaped.shape[0]):
                cov_estimated_reshaped[i,np.where(cov_estimated_reshaped[i,:]==hp.UNSEEN)[0]] = 0.0
            cov_estimated += cov_estimated_reshaped
            '''
            # reorganization of the invAtNA matrix
            # so that it is (n_stokes x n_components )^2 for each sky pixel
            cov_estimated_ = np.zeros(((res.s.shape[0]*res.s.shape[1],res.s.shape[0]*res.s.shape[1], res.s.shape[2])))
            ind0=0
            for c1 in range(res.invAtNA.shape[0]):
                for s1 in range(res.invAtNA.shape[2]):
                    ind1=0
                    for c2 in range(res.invAtNA.shape[1]):
                        res.invAtNA[c1,c2,s1,np.where(res.invAtNA[c1,c2,s1,:]==hp.UNSEEN)[0]] = 0.0
                        for s2 in range(res.invAtNA.shape[2]):
                            if s1==s2: cov_estimated_[ind0,ind1,:] = res.invAtNA[c1,c2,s1,:]*1.0
                            ind1+=1
                    ind0+=1
            cov_estimated += cov_estimated_

        ## SAVING PRODUCTS
        np.save(self.get_output('A_maxL'), A_maxL_v)

        hp.write_map(self.get_output('mask_patches'), mask_patches, overwrite=True)
        hp.write_map(self.get_output('post_compsep_noise'), noise_after_comp_sep_, overwrite=True)

        column_names = []
        [ column_names.extend( (('I_'+str(ch))*optI,('Q_'+str(ch))*optQU,('U_'+str(ch))*optQU)) for ch in A.components]
        column_names = [x for x in column_names if x]
        hp.write_map(self.get_output('post_compsep_maps'), maps_estimated, overwrite=True, column_names=column_names)

        hp.write_map(self.get_output('post_compsep_cov'), cov_estimated.reshape(-1, cov_estimated.shape[-1]), overwrite=True)

        column_names = []
        [column_names.append(str(A.params[i])) for i in range(len(A.params))]
        all_combinations = []
        [all_combinations.append(str(A.params[i])+' x '+str(A.params[j])) for i, j in zip(list(np.triu_indices(len(A.params))[0]),list(np.triu_indices(len(A.params))[1]) )]
        [column_names.append(all_combinations[i]) for i in range(len(all_combinations))]
        if self.config['Nspec']!=0.0:
            column = np.hstack((resx[0],  list(resS[0][np.triu_indices(len(A.params))])))
            for p in range(self.config['Nspec'])[1:]:
                column_ = np.hstack((resx[p],  list(resS[p][np.triu_indices(len(A.params))])))
                column = np.vstack((column, column_))
            np.savetxt(self.get_output('fitted_spectral_parameters'), column, comments=column_names)
        else:
            np.savetxt(self.get_output('fitted_spectral_parameters'), np.hstack((res.x, list(res.Sigma[np.triu_indices(len(A.params))]))), comments=column_names)

if __name__ == '__main__':
    results = PipelineStage.main()

