from bbpipe import PipelineStage
from .types import FitsFile, TextFile
import numpy as np
import pylab as pl
import fgbuster
from fgbuster.cosmology import _get_Cl_cmb, _get_Cl_noise
import healpy as hp
import pymaster as nmt
import sys

class BBREstimation(PipelineStage):
    """
    Stage that estimates the tensor-to-scalar ratio
    (from the power spectrum of the clean CMB map)
    as well as the corresponding error bar (from
    the post comp sep noise covariance)

    The code also performs a marginalization over a 
    provided template of foregrounds residuals (e.g. dust)
    """

    name='BBREstimation'
    inputs=[('Cl_clean', FitsFile),('Cl_cov_clean', FitsFile)]
    outputs=[('estimated_cosmo_params', TextFile)]

    def run(self):

        Cl_clean = hp.read_cl(self.get_input('Cl_clean'))
        Cl_cov_clean = hp.read_cl(self.get_input('Cl_cov_clean'))

        ell_v = Cl_clean[0]        
        
        def from_Cl_to_r_estimate(ClBB_obs, ell_v, fsky, Cl_BB_prim, ClBB_model_other_than_prim, r_v, bins, **minimize_kwargs):

            def likelihood_on_r_computation( r_loc, make_figure=False ):
                '''
                -2logL = sum_ell [ (2l+1)fsky * ( log(C) + C^-1.D  ) ]
                    cf. eg. Tegmark 1998
                '''    
                Cov_model = bins.bin_cell(Cl_BB_prim[:3*self.config['nside']]*r_loc)[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]\
                                            + ClBB_model_other_than_prim

                pl.figure()
                pl.loglog( bins.bin_cell(Cl_BB_prim[:3*self.config['nside']]*r_loc)[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])], label='prim B' )
                pl.loglog( ClBB_model_other_than_prim, label='other than prim B' )
                pl.loglog(ClBB_obs, label='obs BB')
                pl.legend()
                pl.show()

                logL = np.sum( (2*ell_v[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]+1)*fsky\
                                    *( np.log( Cov_model ) + ClBB_obs/Cov_model ))
                return logL

            # gridding -2log(L)
            logL = r_v*0.0
            for ir in range(len(r_v)):
                logL[ir] = likelihood_on_r_computation( r_v[ir] )
                ind = ir*100.0/len(r_v)
                sys.stdout.write("\r  .......... gridding the likelihood on tensor-to-scalar ratio >>>  %d %% " % ind )
                sys.stdout.flush()
            sys.stdout.write("\n")
            # renormalizing logL 
            chi2 = (logL - np.min(logL))
            # computing the likelihood itself, for plotting purposes
            likelihood_on_r = np.exp( - chi2 )/np.max(np.exp( - chi2 ))
            # estimated r is given by:
            r_fit = r_v[np.argmin(logL)]
            if r_fit ==1e-5: r_fit = 0.0
            # and the 1-sigma error bar by (numerical recipies)
            ind_sigma = np.argmin(np.abs( (logL[np.argmin(logL):] - logL[np.argmin(logL)]) - 1.00 ))    
            sigma_r_fit =  r_v[ind_sigma+np.argmin(logL)] - r_fit
            print('NB: sigma(r) is ', sigma_r_fit, ' ( +/- ', r_v[ind_sigma+np.argmin(logL)-1] - r_fit, ' , ', r_v[ind_sigma+np.argmin(logL)+1] - r_fit, ' ) ')
            print('-----')

            return r_fit, sigma_r_fit, likelihood_on_r, chi2


        print('cosmological analysis now ... ')
        ## data first
        lmin = self.config['lmin']
        lmax = self.config['lmax']
        ell_v = Cl_clean[0]#[(ell_v>=lmin)&(ell_v<=lmax)]
        ClBB_obs = Cl_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]
        ClBB_cov_obs = Cl_cov_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]

        # model 
        Cl_BB_lens = _get_Cl_cmb(1.,0.)[2]#[lmin:lmax]
        Cl_BB_prim = _get_Cl_cmb(0.0,self.config['r_input'])[2]#[lmin:lmax]
        bins = nmt.NmtBin(self.config['nside'], nlb=int(1./self.config['fsky']))

        Cl_BB_lens_bin = bins.bin_cell(Cl_BB_lens[:3*self.config['nside']])

        ClBB_model_other_than_prim =  Cl_BB_lens_bin[(ell_v>=lmin)&(ell_v<=lmax)] + Cl_cov_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]

        r_v = np.linspace(-0.001,0.1,num=1000)
        r_v = [0.001,0.01]
        
        r_fit, sigma_r_fit, gridded_likelihood, gridded_chi2 = from_Cl_to_r_estimate(ClBB_obs,
                            ell_v, self.config['fsky'], _get_Cl_cmb(0.,1.)[2],
                                   ClBB_model_other_than_prim, r_v, bins) 
        print('r_fit = ', r_fit)
        print('sigma_r_fit = ', sigma_r_fit)
        column_names = ['r_fit', 'sigma_r']
        np.savetxt(self.get_output('estimated_cosmo_params'), np.hstack((r_fit,  sigma_r_fit)), comments=column_names)

if __name__ == '__main__':
    results = PipelineStage.main()