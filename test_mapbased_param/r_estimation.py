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
    inputs=[('Cl_clean', FitsFile),('Cl_cov_clean', FitsFile), ('Cl_BB_prim_r1', FitsFile), ('Cl_BB_lens', FitsFile)]
    outputs=[('estimated_cosmo_params', TextFile)]

    def run(self):

        Cl_clean = hp.read_cl(self.get_input('Cl_clean'))
        Cl_cov_clean = hp.read_cl(self.get_input('Cl_cov_clean'))

        ell_v = Cl_clean[0]        
        
        def from_Cl_to_r_estimate(ClBB_obs, ell_v, fsky, Cl_BB_prim, ClBB_model_other_than_prim, r_v, bins, Cl_BB_lens_bin, **minimize_kwargs):

            def likelihood_on_r_computation( r_loc, make_figure=False ):
                '''
                -2logL = sum_ell [ (2l+1)fsky * ( log(C) + C^-1.D  ) ]
                    cf. eg. Tegmark 1998
                '''    
                Cov_model = bins.bin_cell(Cl_BB_prim[:3*self.config['nside']]*r_loc)[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]\
                                            + ClBB_model_other_than_prim
                if make_figure:
                    pl.figure()
                    pl.loglog( bins.bin_cell(Cl_BB_prim[:3*self.config['nside']]*r_loc)[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])], label='prim B' )
                    pl.loglog( Cl_BB_lens_bin[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])], label='lensing', linestyle='--'  )
                    pl.loglog( Cl_cov_clean[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])], label='noise post comp sep', linestyle=':')
                    pl.loglog( ClBB_obs, label='obs BB')
                    pl.loglog( Cov_model, label='modeled BB')
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
            if r_fit == 1e-5: r_fit = 0.0
            # and the 1-sigma error bar by (numerical recipies)
            ind_sigma = np.argmin(np.abs( (logL[np.argmin(logL):] - logL[np.argmin(logL)]) - 1.00 ))    
            sigma_r_fit =  r_v[ind_sigma+np.argmin(logL)] - r_fit

            likelihood_on_r_computation( r_fit, make_figure=True )

            print('NB: sigma(r) is ', sigma_r_fit, ' ( +/- ', r_v[ind_sigma+np.argmin(logL)-1] - r_fit, ' , ', r_v[ind_sigma+np.argmin(logL)+1] - r_fit, ' ) ')
            print('-----')

            return r_fit, sigma_r_fit, likelihood_on_r, chi2


        print('cosmological analysis now ... ')
        ## data first
        lmin = self.config['lmin']
        lmax = self.config['lmax']
        ell_v = Cl_clean[0]#[(ell_v>=lmin)&(ell_v<=lmax)]
        ClBB_obs = Cl_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]
        Cl_dust_obs = Cl_clean[2][(ell_v>=lmin)&(ell_v<=lmax)]
        ClBB_cov_obs = Cl_cov_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]

        # model 
        Cl_BB_prim_r1 = hp.read_cl(self.get_input('Cl_BB_prim_r1'))[2]
        Cl_BB_lens = hp.read_cl(self.get_input('Cl_BB_lens'))[2]

        bins = nmt.NmtBin(self.config['nside'], nlb=int(1./self.config['fsky']))

        Cl_BB_lens_bin = bins.bin_cell(self.config['A_lens']*Cl_BB_lens[:3*self.config['nside']])

        if self.config['dust_marginalization']:

            #####################################
            def likelihood_on_r_with_stat_and_sys_res( p_loc ):
                r_loc, A_dust = p_loc 
                Cov_model = bins.bin_cell(Cl_BB_prim[:3*self.config['nside']]*r_loc)[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]\
                                            + ClBB_model_other_than_prim + A_dust*Cl_dust_obs

                logL = np.sum( (2*ell_v[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]+1)*fsky\
                                    *( np.log( Cov_model ) + ClBB_obs/Cov_model ))

                if logL!=logL: 
                    logL = 0.0
                return logL
            
            def lnprior( p_loc ): 
                r_loc, A_stat, A_dust, A_sync, A_dxs = p_loc 
                if 0.0<=r_loc<=1.0 and 1.0<=A_stat<=10.0:
                    return 0.0
                return np.inf

            def lnprob(p_loc):
                lp = 0.0
                return lp + likelihood_on_r_with_stat_and_sys_res(p_loc)

            neg_likelihood_on_r_with_stat_and_sys_res = lambda *args: lnprob(*args)
            pos_likelihood_on_r_with_stat_and_sys_res = lambda *args: -likelihood_on_r_with_stat_and_sys_res(*args)

            ### optimization
            Astat_best_fit_with_stat_res =  scipy.optimize.minimize( pos_likelihood_on_r_with_stat_and_sys_res, \
                    [1.0,0.1],\
                    tol=1e-18, method='TNC', \
                    bounds=[(0.0, 1e2), (0.0, None)],\
                    options={'disp':True, 'gtol': 1e-18, 'eps': 1e-6,\
                    'maxiter':1000, 'ftol': 1e-18})

            print '#'*20
            print '#'*20
            print '################### result of the optimization ... '
            print Astat_best_fit_with_stat_res['x']

            ### sampling
            import emcee
            ndim, nwalkers = self.config['ndim'], self.config['nwalkers']
            p0 = [np.random.rand(ndim) for i in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, neg_likelihood_on_r_with_stat_and_sys_res)
            sampler.run_mcmc(p0, 10000)

            samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
            truths = []
            for i in range(len(Astat_best_fit_with_stat_res['x'])):
                truths.append(Astat_best_fit_with_stat_res['x'][i])

            ### plotting 
            import getdist
            from getdist import plots, MCSamples
            pl.rcParams['text.usetex']=False
            # ci-dessous names et labels definient les parametres du corner plot
            names = ["r", "\Lambda_d",]
            labels =  ["r", "\Lambda_d"]
            g = plots.getSubplotPlotter()
            samples = ## ce qui sort du emcee
            samps = MCSamples(samples=samples, names=names, labels=labels)

            g.triangle_plot(samps, filled=True)#,
                # legend_labels=legend_labels, line_args=[{'lw':2,'color':color_loc[0],'alpha':0.7},{'lw':2,'color':color_loc[1],'alpha':0.7}])

            pl.show()

        else:
            #####################################
            ClBB_model_other_than_prim = Cl_BB_lens_bin[(ell_v>=lmin)&(ell_v<=lmax)]\
                                 + Cl_cov_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]

            r_v = np.logspace(-5,0,num=1000)

            r_fit, sigma_r_fit, gridded_likelihood, gridded_chi2 = from_Cl_to_r_estimate(ClBB_obs,
                                ell_v, self.config['fsky'], Cl_BB_prim_r1,
                                       ClBB_model_other_than_prim, r_v, bins, Cl_BB_lens_bin)
            pl.figure()
            pl.semilogx(r_v, gridded_likelihood)
            pl.show()

        print('r_fit = ', r_fit)
        print('sigma_r_fit = ', sigma_r_fit)
        column_names = ['r_fit', 'sigma_r']
        np.savetxt(self.get_output('estimated_cosmo_params'), np.hstack((r_fit,  sigma_r_fit)), comments=column_names)

if __name__ == '__main__':
    results = PipelineStage.main()

