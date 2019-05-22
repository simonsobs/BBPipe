from bbpipe import PipelineStage
from .types import FitsFile, TextFile, PdfFile, NumpyFile
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl
import fgbuster
from fgbuster.cosmology import _get_Cl_cmb, _get_Cl_noise
import healpy as hp
import pymaster as nmt
import sys
import scipy 
from .Cl_estimation import binning_definition
from fgbuster.algebra import W_dB, _mmm
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.mixingmatrix import MixingMatrix

def Cl_stat_model(Cl_fgs, Sigma, components, instrument, beta_maxL, invN, i_cmb=0):
    """
    This function estimates the statistical foregrounds
    residuals from the input frequency cross spectra, Cl_fgs, 
    and the error bar covariance, Sigma
    """
    A = MixingMatrix(*components)
    A_ev = A.evaluator(instrument['frequencies'])
    # from the mixing matrix and its derivative at the peak
    # of the likelihood, we build dW/dB
    A_maxL = A_ev(beta_maxL[:,i_patch])
    A_dB_maxL = A_dB_ev(beta_maxL[:,i_patch])
    # patch_invN = _indexed_matrix(invN, d_spectra.T.shape, patch_mask)
    comp_of_dB = A.comp_of_dB
    W_dB_maxL = W_dB(A_maxL, A_dB_maxL, comp_of_dB, invN=None)[:, i_cmb]
    # and then Cl_YY, cf Stompor et al 2016
    Cl_YY = _mmm(W_dB_maxL, Cl_fgs.T, W_dB_maxL.T)  
    # and finally, using the covariance of error bars on spectral indices
    # we compute the model for the statistical foregrounds residuals, 
    # cf. Errard et al 2018
    tr_SigmaYY = np.einsum('ij, lji -> l', Sigma, Cl_YY)
    Cl_stat_res_model = tr_SigmaYY

    return Cl_stat_res_model


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
    inputs=[('Cl_clean', FitsFile),('Cl_noise', FitsFile),('Cl_cov_clean', FitsFile), ('Cl_BB_prim_r1', FitsFile), 
                ('Cl_BB_lens', FitsFile), ('fsky_eff',TextFile), ('Cl_fgs', NumpyFile), 
                    ('fitted_spectral_parameters', TextFile)]
    outputs=[('estimated_cosmo_params', TextFile), ('likelihood_on_r', PdfFile), 
                ('power_spectrum_post_comp_sep', PdfFile), ('gridded_likelihood', NumpyFile)]

    def run(self):

        Cl_clean = hp.read_cl(self.get_input('Cl_clean'))
        Cl_noise = hp.read_cl(self.get_input('Cl_noise'))
        Cl_cov_clean = hp.read_cl(self.get_input('Cl_cov_clean'))
        fsky_eff = np.loadtxt(self.get_input('fsky_eff'))
        ell_v = Cl_clean[0]        
        
        print('cosmological analysis now ... ')
        ## data first
        lmin = self.config['lmin']
        lmax = self.config['lmax']
        ell_v = Cl_clean[0]
        ClBB_obs = Cl_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]
        # pl.figure()
        # pl.loglog(ClBB_obs)
        # pl.loglog(Cl_noise[1][(ell_v>=lmin)&(ell_v<=lmax)], ':')
        # pl.loglog(Cl_cov_clean[1][(ell_v>=lmin)&(ell_v<=lmax)], '--')
        # pl.show()
        # exit()
        Cl_dust_obs = Cl_clean[2][(ell_v>=lmin)&(ell_v<=lmax)]- Cl_noise[2][(ell_v>=lmin)&(ell_v<=lmax)]
        Cl_sync_obs = Cl_clean[3][(ell_v>=lmin)&(ell_v<=lmax)]- Cl_noise[3][(ell_v>=lmin)&(ell_v<=lmax)]
        ClBB_cov_obs = Cl_cov_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]

        ################ STATISTICAL FOREGROUNDS RESIDUALS MODELING
        Cl_fgs = np.load(self.get_input('Cl_fgs'))
        p = np.loadtxt(self.get_input('fitted_spectral_parameters'))
        ## the length of p is always n_params  + (nparams*(nparams+1)/2)
        ## = nparams + (nparams**2/2 + nparams/2)
        ## = nparams**2 /2 + nparams*3/2
        ## it would be nice to have this not hardcoded eventually 
        beta_maxL = p[:2]
        Sigma = np.array([[p[2],p[3]],[p[3],p[4]]])
        instrument = {'frequencies':np.array(self.config['frequencies'])}
        components = [CMB(), Dust(150., temp=20.0), Synchrotron(150.)]
        Cl_stat_res_model = Cl_stat_model(Cl_fgs, components, instrument, beta_maxL, None, i_cmb=0)
        ################

        # model 
        Cl_BB_prim_r1 = hp.read_cl(self.get_input('Cl_BB_prim_r1'))[2]
        Cl_BB_lens = hp.read_cl(self.get_input('Cl_BB_lens'))[2]
        
        bins = binning_definition(self.config['nside'], lmin=self.config['lmin'], lmax=self.config['lmax'],\
                             nlb=self.config['nlb'], custom_bins=self.config['custom_bins'])
        # bins = nmt.NmtBin(self.config['nside'], nlb=int(1./self.config['fsky']))

        Cl_BB_lens_bin = bins.bin_cell(self.config['A_lens']*Cl_BB_lens[:3*self.config['nside']])
        ClBB_model_other_than_prim = Cl_BB_lens_bin[(ell_v>=lmin)&(ell_v<=lmax)]

        if self.config['noise_option']!='no_noise': 
            ClBB_model_other_than_prim += Cl_cov_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]
            # ClBB_model_other_than_prim += Cl_noise[1][(ell_v>=lmin)&(ell_v<=lmax)]
        if self.config['include_stat_res']:
            ClBB_model_other_than_prim += Cl_stat_res_model[(ell_v>=lmin)&(ell_v<=lmax)]

        if self.config['dust_marginalization']:

            #####################################
            def likelihood_on_r_with_stat_and_sys_res( p_loc, bins=bins, make_figure=False ):
                if self.config['sync_marginalization']:
                    r_loc, A_dust, A_sync = p_loc 
                else:
                    r_loc, A_dust = p_loc 
                Cov_model = bins.bin_cell(Cl_BB_prim_r1[:3*self.config['nside']]*r_loc)[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]\
                                            + ClBB_model_other_than_prim + A_dust*Cl_dust_obs

                if self.config['sync_marginalization']: 
                    Cov_model += A_sync*Cl_sync_obs

                if make_figure:
                    print('actual noise after comp sep = ', Cl_noise[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])])

                    # pl.loglog( ell_v_loc, norm*bins.bin_cell(Cl_BB_prim[:3*self.config['nside']]*r_loc)[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                    #              label='primordial BB, r = '+str(r_loc), linestyle='--', color='Purple', linewidth=2.0 )
                    # pl.loglog( ell_v_loc, norm*Cl_BB_lens_bin[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])], 
                    #             label='lensing BB', linestyle='-', color='DarkOrange', linewidth=2.0)
                    # pl.loglog( ell_v_loc, norm*Cl_cov_clean[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])], 
                    #             label='noise post comp sep', linestyle=':', color='DarkBlue')
                    # pl.loglog( ell_v_loc, norm*ClBB_obs, label='observed BB', color='red', linestyle='-', linewidth=2.0, alpha=0.8)
                    # pl.loglog( ell_v_loc, norm*Cov_model, label='modeled BB', color='k', linestyle='-', linewidth=2.0, alpha=0.8)

                    pl.figure(num=None, figsize=(14,10), facecolor='w', edgecolor='k')
                    ell_v_loc = ell_v[(ell_v>=lmin)&(ell_v<=lmax)]
                    norm = ell_v_loc*(ell_v_loc+1)/2/np.pi
                    pl.loglog( ell_v_loc, norm*bins.bin_cell(Cl_BB_prim_r1[:3*self.config['nside']]*r_loc)[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                                  label='primordial BB, r = '+str(r_loc), linestyle='--', color='Purple', linewidth=2.0 )
                    pl.loglog( ell_v_loc, norm*Cl_BB_lens_bin[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                                 label='lensing BB', linestyle='-', color='DarkOrange', linewidth=2.0)
                    pl.loglog( ell_v_loc, norm*Cl_cov_clean[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                                 label='estimated noise post comp sep', linestyle=':', color='DarkBlue')
                    pl.loglog( ell_v_loc, norm*Cl_noise[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                                 label='actual noise post comp sep', linestyle=':', color='Cyan')
                    pl.loglog( ell_v_loc, norm*Cl_noise[2][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                                 label='actual dust noise post comp sep', linestyle=':', color='DarkGray')
                    pl.loglog( ell_v_loc, norm*Cl_dust_obs, label='estimated dust template', linestyle='-', color='DarkGray', linewidth=2.0, alpha=0.8)
                    pl.loglog( ell_v_loc, norm*A_dust*Cl_dust_obs, label='bias-fitted dust template', linestyle='--', color='DarkGray', linewidth=2.0, alpha=0.8)
                    if self.config['sync_marginalization']: pl.loglog( ell_v_loc, norm*A_sync*Cl_sync_obs,
                                                 label='synchrotron template', linestyle='--', color='DarkGray', linewidth=2.0, alpha=0.8)
                    pl.loglog( ell_v_loc, norm*(ClBB_obs - Cl_noise[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]), label='observed BB - actual noise', color='red', linestyle='-', linewidth=2.0, alpha=0.8)
                    pl.loglog( ell_v_loc, norm*(Cov_model - Cl_cov_clean[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]), label='modeled BB - modeled noise', color='k', linestyle='-', linewidth=2.0, alpha=0.8)
                    if self.config['include_stat_res']: pl.loglog( ell_v_loc, norm*Cl_stat_res_model[(ell_v>=self.config['lmin'])
                                            &(ell_v<=self.config['lmax'])], label='modeled stat residuals', color='r', linestyle='--',
                                                 linewidth=2.0, alpha=0.8)
                    ax = pl.gca()
                    box = ax.get_position()
                    ax.set_position([box.x0-box.width*0.02, box.y0, box.width*0.8, box.height])
                    # ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
                    legend = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size':10}, ncol=1, labelspacing=1.0)
                    frame = legend.get_frame()
                    frame.set_edgecolor('white')
                    # pl.legend()
                    pl.xlabel('$\ell$', fontsize=20)
                    pl.ylabel('$D_\ell$ $[\mu K^2]$', fontsize=20)
                    pl.ylim([1e-5,2e-1])
                    pl.savefig(self.get_output('power_spectrum_post_comp_sep'))
                    # pl.show()
                    pl.close()

                logL = 0.0
                for b in range(len(ClBB_obs)):
                    logL -= np.sum((2*bins.get_ell_list(b)+1))*fsky_eff/2*( np.log( Cov_model[b] ) + ClBB_obs[b]/Cov_model[b] )
                if logL!=logL: 
                    logL = 0.0
                return logL
            
            def lnprior( p_loc ): 
                r_loc, A_dust = p_loc 
                # if -5e-3<=r_loc  and 0.0<=A_dust:
                    # return 0.0
                return -np.inf

            def lnprob(p_loc):
                # lp = 0.0
                lp = lnprior( p_loc )
                return lp + likelihood_on_r_with_stat_and_sys_res(p_loc)

            neg_likelihood_on_r_with_stat_and_sys_res = lambda *args: lnprob(*args)
            pos_likelihood_on_r_with_stat_and_sys_res = lambda *args: -likelihood_on_r_with_stat_and_sys_res(*args)

            ### optimization
            if self.config['sync_marginalization']:
                bounds = [(0.0, None), (0.0, None), (0.0, None)]
                p0 = [1.0,0.1,0.1]
                names = ["r", "\Lambda_d", "\Lambda_s"]
                labels =  ["r", "\Lambda_d", "\Lambda_s"]
            else:
                # bounds = [(0.0, None), (0.0, None)]
                bounds = [(None, None), (None, None)]
                p0 = [1.0,0.1]
                names = ["r", "\Lambda_d",]
                labels =  ["r", "\Lambda_d"]

            Astat_best_fit_with_stat_res =  scipy.optimize.minimize( pos_likelihood_on_r_with_stat_and_sys_res, \
                    p0,\
                    tol=1e-18, method='TNC', \
                    bounds=bounds,\
                    options={'disp':True, 'gtol': 1e-18, 'eps': 1e-6,\
                    'maxiter':1000, 'ftol': 1e-18})

            print('#'*20)
            print('################### result of the optimization ... ')
            print(Astat_best_fit_with_stat_res['x'])

            ### making plot after optimization
            likelihood_on_r_with_stat_and_sys_res( Astat_best_fit_with_stat_res['x'], make_figure=True )

            ### sampling
            import emcee
            ndim, nwalkers = self.config['ndim'], self.config['nwalkers']
            p0 = [np.random.rand(ndim) for i in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, neg_likelihood_on_r_with_stat_and_sys_res)#, threads=4)
            sampler.run_mcmc(p0, 10000)

            samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))
            truths = []
            for i in range(len(Astat_best_fit_with_stat_res['x'])):
                truths.append(Astat_best_fit_with_stat_res['x'][i])

            ### plotting 
            import getdist
            from getdist import plots, MCSamples
            pl.rcParams['text.usetex']=False
            # ci-dessous names et labels definient les parametres du corner plot

            g = plots.getSubplotPlotter()
            samps = MCSamples(samples=samples, names=names, labels=labels)

            g.triangle_plot(samps, filled=True)#,
                # legend_labels=legend_labels, line_args=[{'lw':2,'color':color_loc[0],'alpha':0.7},{'lw':2,'color':color_loc[1],'alpha':0.7}])
            # pl.savefig('./test_sampling_r_Adust.pdf')
            pl.savefig(self.get_output('likelihood_on_r'))
            pl.close()
            ##############
            # print(samps.getInlineLatex('r',limit=1))
            # print(samps.getMeans())
            # print(samps.getVars())
            ##############
            r_fit = samps.getMeans()[names.index("r")]
            sigma_r_fit = np.sqrt(samps.getVars()[names.index("r")])

            # pl.show()
            # exit()

        else:

            def from_Cl_to_r_estimate(ClBB_obs, ell_v, Cl_BB_prim, ClBB_model_other_than_prim, r_v, bins, Cl_BB_lens_bin, **minimize_kwargs):

                def likelihood_on_r_computation( r_loc, make_figure=False ):
                    '''
                    -2logL = sum_ell [ (2l+1)fsky * ( log(C) + C^-1.D  ) ]
                        cf. eg. Tegmark 1998
                    '''    
                    Cov_model = bins.bin_cell(Cl_BB_prim[:3*self.config['nside']]*r_loc)[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]\
                                                + ClBB_model_other_than_prim
                    if make_figure:
                        pl.figure( figsize=(10,7), facecolor='w', edgecolor='k' )
                        ell_v_loc = ell_v[(ell_v>=lmin)&(ell_v<=lmax)]
                        norm = ell_v_loc*(ell_v_loc+1)/2/np.pi
                        pl.loglog( ell_v_loc, norm*bins.bin_cell(Cl_BB_prim[:3*self.config['nside']]*r_loc)[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                     label='primordial BB, r = '+str(r_loc), linestyle='--', color='Purple', linewidth=2.0 )
                        pl.loglog( ell_v_loc, norm*Cl_BB_lens_bin[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])], 
                                    label='lensing BB', linestyle='-', color='DarkOrange', linewidth=2.0)
                        pl.loglog( ell_v_loc, norm*Cl_cov_clean[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])], 
                                    label='noise post comp sep', linestyle=':', color='DarkBlue')
                        pl.loglog( ell_v_loc, norm*Cl_noise[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                                     label='actual CMB noise post comp sep', linestyle=':', color='Cyan')
                        pl.loglog( ell_v_loc, norm*Cl_noise[2][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                                     label='actual dust noise post comp sep', linestyle=':', color='DarkGray')
                        pl.loglog( ell_v_loc, norm*Cl_dust_obs, label='estimated dust template', linestyle='-', color='DarkGray', linewidth=2.0, alpha=0.8)
                        
                        pl.loglog( ell_v_loc, norm*ClBB_obs, label='observed BB', color='red', linestyle='-', linewidth=2.0, alpha=0.8)
                        pl.loglog( ell_v_loc, norm*Cov_model, label='modeled BB', color='k', linestyle='-', linewidth=2.0, alpha=0.8)
                        pl.legend()
                        pl.xlabel('$\ell$', fontsize=20)
                        pl.ylabel('$D_\ell$ $[\mu K^2]$', fontsize=20)
                        pl.savefig(self.get_output('power_spectrum_post_comp_sep'))
                        pl.close()

                    # logL = np.sum( (2*ell_v[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]+1)*fsky\
                                        # *( np.log( Cov_model ) + ClBB_obs/Cov_model ))

                    logL = 0.0 
                    for b in range(len(ClBB_obs)):
                        logL += np.sum((2* bins.get_ell_list(b)+1))*fsky_eff/2*( np.log( Cov_model[b] ) + ClBB_obs[b]/Cov_model[b] )
                    return logL

                # gridding -2log(L)
                logL = r_v*0.0
                for ir in range(len(r_v)):
                    logL[ir] = likelihood_on_r_computation( r_v[ir] )
                    # ind = ir*100.0/len(r_v)
                    # sys.stdout.write("\r  .......... gridding the likelihood on tensor-to-scalar ratio >>>  %d %% " % ind )
                    # sys.stdout.flush()
                # sys.stdout.write("\n")
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

            r_v = np.logspace(-5,0,num=1000)

            r_fit, sigma_r_fit, gridded_likelihood, gridded_chi2 = from_Cl_to_r_estimate(ClBB_obs,
                                ell_v, Cl_BB_prim_r1,
                                       ClBB_model_other_than_prim, r_v, bins, Cl_BB_lens_bin)
            pl.figure()
            pl.semilogx(r_v, gridded_likelihood)
            pl.savefig(self.get_output('likelihood_on_r'))
            # pl.show()

        print('r_fit = ', r_fit)
        print('sigma_r_fit = ', sigma_r_fit)
        column_names = ['r', 'L(r)']
        np.savetxt(self.get_output('estimated_cosmo_params'), np.hstack((r_fit,  sigma_r_fit)), comments=column_names)
        if self.config['dust_marginalization'] is False and self.config['sync_marginalization'] is False:
            np.save(self.get_output('gridded_likelihood'), np.hstack((r_v,  gridded_likelihood)))

if __name__ == '__main__':
    results = PipelineStage.main()

