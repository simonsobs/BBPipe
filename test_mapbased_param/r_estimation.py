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
from matplotlib import ticker
from matplotlib.ticker import LogLocator

def ticks_format(value, index):
    """
    get the value and returns the value as:
        integer: [0,99]
        1 digit float: [0.1, 0.99]
        n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:   
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

def Cl_stat_model(Cl_fgs, Sigma, components, instrument, beta_maxL, invN=None, i_cmb=0):
    """
    This function estimates the statistical foregrounds
    residuals from the input frequency cross spectra, Cl_fgs, 
    and the error bar covariance, Sigma
    """
    A = MixingMatrix(*components)
    A_ev = A.evaluator(instrument['frequencies'])
    A_dB_ev = A.diff_evaluator(instrument['frequencies'])
    # from the mixing matrix and its derivative at the peak
    # of the likelihood, we build dW/dB
    A_maxL = A_ev(beta_maxL)
    A_dB_maxL = A_dB_ev(beta_maxL)
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
                    ('fitted_spectral_parameters', TextFile), ('Cl_CMB_template_150GHz', NumpyFile),
                        ('Cl_cov_freq', FitsFile)]
    outputs=[('estimated_cosmo_params', TextFile), ('likelihood_on_r', PdfFile), 
                ('power_spectrum_post_comp_sep', PdfFile), ('gridded_likelihood', NumpyFile), ('power_spectrum_post_comp_sep_v2', PdfFile)]

    def run(self):

        Cl_clean = hp.read_cl(self.get_input('Cl_clean'))
        Cl_noise = hp.read_cl(self.get_input('Cl_noise'))
        Cl_cov_clean = hp.read_cl(self.get_input('Cl_cov_clean'))
        Cl_cov_freq = hp.read_cl(self.get_input('Cl_cov_freq'))
        fsky_eff = np.loadtxt(self.get_input('fsky_eff'))
        ell_v = Cl_clean[0]        
        
        print('cosmological analysis now ... ')
        ## data first
        lmin = self.config['lmin']
        lmax = self.config['lmax']
        ell_v = Cl_clean[0]
        ClBB_obs = Cl_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]

        Cl_dust_obs = Cl_clean[2][(ell_v>=lmin)&(ell_v<=lmax)]- Cl_noise[2][(ell_v>=lmin)&(ell_v<=lmax)]
        Cl_sync_obs = Cl_clean[3][(ell_v>=lmin)&(ell_v<=lmax)]- Cl_noise[3][(ell_v>=lmin)&(ell_v<=lmax)]
        ClBB_cov_obs = Cl_cov_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]

        ################ STATISTICAL FOREGROUNDS RESIDUALS MODELING
        Cl_fgs = np.load(self.get_input('Cl_fgs'))
        Cl_CMB_template_150GHz = np.load(self.get_input('Cl_CMB_template_150GHz'))
        p = np.loadtxt(self.get_input('fitted_spectral_parameters'))
        ## the length of p is always n_params  + (nparams*(nparams+1)/2)
        ## = nparams + (nparams**2/2 + nparams/2)
        ## = nparams**2 /2 + nparams*3/2
        ## it would be nice to have this not hardcoded eventually 
        beta_maxL = p[:2]
        Sigma = np.array([[p[2],p[3]],[p[3],p[4]]])
        instrument = {'frequencies':np.array(self.config['frequencies'])}
        components = [CMB(), Dust(150., temp=20.0), Synchrotron(150.)]
        Cl_stat_res_model = Cl_stat_model(Cl_fgs, Sigma, components, instrument, beta_maxL, None, i_cmb=0)
        ################

        # model 
        Cl_BB_prim_r1 = hp.read_cl(self.get_input('Cl_BB_prim_r1'))[2]
        Cl_BB_lens = hp.read_cl(self.get_input('Cl_BB_lens'))[2]
        
        bins = binning_definition(self.config['nside'], lmin=self.config['lmin'], lmax=self.config['lmax'],\
                             nlb=self.config['nlb'], custom_bins=self.config['custom_bins'])
        ell_v_eff = bins.get_effective_ells()
        # bins = nmt.NmtBin(self.config['nside'], nlb=int(1./self.config['fsky']))

        # Cl_BB_lens_bin = bins.bin_cell(self.config['A_lens']*Cl_BB_lens[2:3*self.config['nside']+2])
        Cl_BB_lens_bin = bins.bin_cell(self.config['A_lens']*Cl_BB_lens[:3*self.config['nside']])
        ClBB_model_other_than_prim = Cl_BB_lens_bin[(ell_v>=lmin)&(ell_v<=lmax)]
        ClBB_model_other_than_prim_and_lens = Cl_BB_lens_bin[(ell_v>=lmin)&(ell_v<=lmax)]*0.0

        if self.config['noise_option']!='no_noise': 
            ClBB_model_other_than_prim += Cl_cov_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]
            ClBB_model_other_than_prim_and_lens += Cl_cov_clean[1][(ell_v>=lmin)&(ell_v<=lmax)]
            # ClBB_model_other_than_prim += Cl_noise[1][(ell_v>=lmin)&(ell_v<=lmax)]
        if self.config['include_stat_res']:
            ClBB_model_other_than_prim += Cl_stat_res_model[(ell_v>=lmin)&(ell_v<=lmax)]
            ClBB_model_other_than_prim_and_lens += Cl_stat_res_model[(ell_v>=lmin)&(ell_v<=lmax)]

        if self.config['dust_marginalization']:

            #####################################
            def likelihood_on_r_with_stat_and_sys_res( p_loc, bins=bins, make_figure=False, tag='' ):
                if self.config['sync_marginalization']:
                    if self.config['AL_marginalization']:
                        r_loc, A_dust, A_sync, AL = p_loc 
                    else:
                        r_loc, A_dust, A_sync = p_loc 
                        AL = 1.0
                else:
                    if self.config['AL_marginalization']:
                        r_loc, A_dust, AL = p_loc
                    else:
                        r_loc, A_dust = p_loc
                        AL = 1.0

                Cov_model = bins.bin_cell(Cl_BB_prim_r1[:3*self.config['nside']]*r_loc)[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]\
                                            + ClBB_model_other_than_prim_and_lens + A_dust*Cl_dust_obs + AL*Cl_BB_lens_bin[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]

                if self.config['sync_marginalization']: 
                    Cov_model += A_sync*Cl_sync_obs

                if make_figure:
                    # print('actual noise after comp sep = ', Cl_noise[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])])

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

                    ell_v_loc_eff = ell_v_eff[(ell_v_eff>=lmin)&(ell_v_eff<=lmax)]
                    norm_eff = ell_v_loc_eff*(ell_v_loc_eff+1)/2/np.pi
                    # theory BB primordial
                    pl.loglog( ell_v_loc_eff, norm*bins.bin_cell(Cl_BB_prim_r1[:3*self.config['nside']]*r_loc)[(ell_v_loc_eff>=self.config['lmin'])&(ell_v_loc_eff<=self.config['lmax'])],
                                                label='primordial BB, r = '+str(r_loc), linestyle='--', color='Purple', linewidth=2.0 )
                    # theory lensing
                    pl.loglog( ell_v_loc_eff, norm*Cl_BB_lens_bin[(ell_v_loc_eff>=self.config['lmin'])&(ell_v_loc_eff<=self.config['lmax'])],
                                                label='lensing BB', linestyle='-', color='DarkOrange', linewidth=2.0)
                    if self.config['AL_marginalization']: pl.loglog( ell_v_loc, norm*AL*Cl_BB_lens_bin[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                                label='fitted lensing BB', linestyle='--', color='DarkOrange', linewidth=2.0)
                    # estimated noise bias on CMB
                    pl.loglog( ell_v_loc, norm*Cl_cov_clean[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                                label='estimated noise post comp sep', linestyle=':', color='DarkBlue')
                    # true noise bias on CMB
                    pl.loglog( ell_v_loc, norm*Cl_noise[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                                label='actual noise post comp sep', linestyle=':', color='Cyan')
                    # true noise bias - observed noise bias 
                    # pl.loglog( ell_v_loc, norm*np.abs(Cl_noise[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]-Cl_cov_clean[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]),
                                                # label='noise difference', linestyle=':', color='purple')
                    # true noise bias on dust
                    # pl.loglog( ell_v_loc, norm*Cl_noise[2][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])],
                                                # label='actual dust noise post comp sep', linestyle=':', color='DarkGray')
                    # estimated dust template
                    pl.loglog( ell_v_loc, norm*Cl_dust_obs, label='estimated dust template @ 150GHz', linestyle='-', color='DarkGray', linewidth=2.0, alpha=0.8)
                    # rescaled dust template to mimic foregrounds residuals
                    pl.loglog( ell_v_loc, norm*A_dust*Cl_dust_obs, label='bias-fitted dust template', linestyle='--', color='DarkGray', linewidth=2.0, alpha=0.8)
                    # same for synchrotron
                    if self.config['sync_marginalization']: pl.loglog( ell_v_loc, norm*A_sync*Cl_sync_obs,
                                                label='synchrotron template @ 150GHz', linestyle='--', color='DarkGray', linewidth=2.0, alpha=0.8)
                    # true noise-debiased BB spectrum
                    # which should correspond to primordial BB + lensing BB + foregrounds residuals
                    pl.loglog( ell_v_loc, norm*(ClBB_obs - Cl_noise[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]), 
                                                label='observed BB - actual noise = tot BB + residuals', 
                                                color='red', linestyle='-', linewidth=2.0, alpha=0.8)
                    # modeled noise-debiased BB spectrum
                    # which should correspond to primordial BB + lensing BB + foregrounds residuals
                    pl.loglog( ell_v_loc, norm*(Cov_model - Cl_cov_clean[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]), 
                                                label='modeled BB - modeled noise = tot BB + residuals', 
                                                color='k', linestyle='-', linewidth=2.0, alpha=0.8)

                    # pl.loglog( ell_v_loc, norm*(ClBB_obs - Cov_model), 
                                                # label='observed BB - modeled BB', 
                                                # color='red', linestyle=':', linewidth=2.0, alpha=0.8)

                    # including statistical foregrounds residuals
                    if self.config['include_stat_res']: pl.loglog( ell_v_loc, norm*Cl_stat_res_model[(ell_v>=self.config['lmin'])
                                            &(ell_v<=self.config['lmax'])], label='modeled stat residuals', color='r', linestyle='--',
                                                linewidth=2.0, alpha=0.8)
                    # including true CMB template @ 150GHz
                    pl.loglog( ell_v_loc, norm*Cl_CMB_template_150GHz[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])], linestyle=':', color='red', 
                                                linewidth=3.0, alpha=1.0, label='input CMB template @ 150GHz')
                    # noise per frequency channel  
                    for i in range(len(Cl_cov_freq)):
                        print(norm*Cl_cov_freq[i][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])])
                        pl.loglog( ell_v_loc, norm*Cl_cov_freq[i][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])], 
                                            linestyle='-', color='cyan', linewidth=3.0, alpha=0.8, label='noise for frequency '+str(i))
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
                    pl.ylim([1e-6,2e-1])

                    subs = [ 1.0, 2.0, 5.0 ]  
                    ax.xaxis.set_major_locator( ticker.LogLocator( subs=subs ) )
                    ax.xaxis.set_minor_locator( ticker.LogLocator( subs=subs ) ) #set the ticks position
                    ax.xaxis.set_major_formatter( ticker.NullFormatter() )   # remove the major ticks
                    ax.xaxis.set_minor_formatter( ticker.FuncFormatter(ticks_format) )
                    ax.yaxis.set_major_locator( ticker.LogLocator( subs=subs ) )
                    ax.yaxis.set_minor_locator( ticker.LogLocator( subs=subs ) ) #set the ticks position
                    ax.yaxis.set_major_formatter( ticker.NullFormatter() )   # remove the major ticks
                    ax.yaxis.set_minor_formatter( ticker.FuncFormatter(ticks_format) )

                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(16)
                    for tick in ax.xaxis.get_minor_ticks():
                        tick.label.set_fontsize(16)
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(16)
                    for tick in ax.yaxis.get_minor_ticks():
                        tick.label.set_fontsize(16)

                    pl.savefig(self.get_output('power_spectrum_post_comp_sep'+tag))
                    # pl.show()
                    pl.close()

                logL = 0.0
                for b in range(len(ClBB_obs)):
                    logL -= np.sum((2*bins.get_ell_list(b)+1))*fsky_eff/2*( np.log( Cov_model[b] ) + ClBB_obs[b]/Cov_model[b] )
                if logL!=logL: 
                    logL = 0.0
                return logL
            
            def lnprior( p_loc ): 
                if self.config['sync_marginalization']:
                    if self.config['AL_marginalization']:
                        r_loc, A_dust, A_sync, AL = p_loc 
                    else:
                        r_loc, A_dust, A_sync = p_loc 
                else:
                    if self.config['AL_marginalization']:
                        r_loc, A_dust, AL = p_loc 
                    else:
                        r_loc, A_dust = p_loc 
                if 0.0>r_loc or 0.0>A_dust:
                # if 0.0>A_dust:
                    return -np.inf
                else: return 0.0
                # if -1e-3<=r_loc  and 
                # if 0.0<=A_dust:
                    # return 0.0
                # print('r = ', r_loc, ' and A_dust = ', A_dust)
                # return -np.inf

            def lnprob(p_loc):
                # lp = 0.0
                lp = lnprior( p_loc )
                # print('lp = ', lp)
                return lp + likelihood_on_r_with_stat_and_sys_res(p_loc)

            neg_likelihood_on_r_with_stat_and_sys_res = lambda *args: lnprob(*args)
            pos_likelihood_on_r_with_stat_and_sys_res = lambda *args: -likelihood_on_r_with_stat_and_sys_res(*args)

            ### optimization
            if self.config['sync_marginalization']:
                if self.config['AL_marginalization']:
                    bounds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None)]
                    p0 = [1.0,0.1,0.1, 1.0]
                    names = ["r", "\Lambda_d", "\Lambda_s", "A_L"]
                    labels =  ["r", "\Lambda_d", "\Lambda_s", "A_L"]
                else:
                    bounds = [(0.0, None), (0.0, None), (0.0, None)]
                    p0 = [1.0,0.1,0.1]
                    names = ["r", "\Lambda_d", "\Lambda_s"]
                    labels =  ["r", "\Lambda_d", "\Lambda_s"]
            else:
                if self.config['AL_marginalization']:
                    bounds = [(0.0, None), (0.0, None), (0.0, None)]
                    p0 = [1.0, 0.1, 1.0]
                    names = ["r", "\Lambda_d", "A_L"]
                    labels =  ["r", "\Lambda_d", "A_L"]
                else:
                    bounds = [(0.0, None), (0.0, None)]
                    # bounds = [(-0.01, None), (0.0, None)]
                    # bounds = [(None, None), (None, None)]
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
            sampler.run_mcmc(p0, 5000)

            samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
            

            ######################################
            for p in ['r', 'Ad']:
                if p == 'r': ind = 0
                else : ind = 1
                counts, bins, patches = pl.hist(samples[:,ind], 500)
                pl.close()
                bins_av = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
                ind_r_fit = np.argmax(counts)
                if p == 'r':
                    r_fit = bins_av[ind_r_fit]
                else:
                    Ad_fit = bins_av[ind_r_fit]
                sum_ = 0.0
                sum_tot = 0.0
                for i in range(len(counts))[ind_r_fit:]:
                    sum_tot += counts[i]*(bins[i+1] - bins[i])

                for i in range(len(counts))[ind_r_fit:]:
                    sum_ += counts[i]*(bins[i+1] - bins[i])
                    if sum_ > 0.68*sum_tot:
                        break
                # sigma(r) is the distance between peak and 68% of the integral
                if p == 'r':
                    sigma_r_fit = bins_av[i-1] - r_fit
                else:
                    sigma_Ad_fit = bins_av[i-1] - Ad_fit
            ########################################


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

            ##############
            # print(samps.getInlineLatex('r',limit=1))
            # print(samps.getMeans())
            # print(samps.getVars())
            # ##############
            # r_fit = samps.getMeans()[names.index("r")]
            # # r_fit2 = samps.getBestFit()[names.index("r")]
            # Ad_fit = samps.getMeans()[names.index("\Lambda_d")]
            # # Ad_fit2 = samps.getBestFit()[names.index("\Lambda_d")]
            # sigma_r_fit = np.sqrt(samps.getVars()[names.index("r")])
            # sigma_Ad_fit = np.sqrt(samps.getVars()[names.index("\Lambda_d")])

            #### another way of estimating error bars ..... 
            # print('samples = ', samples)
            # print('w/ shape = ', samples.shape)
            # print('nbins = ', 500)
            # print('r_fit = ', r_fit)
            # print('sigma_r_fit = ', sigma_r_fit)
            # exit()
            ########


            # draw vertical and horizontal lines to display the input and fitted values 
            # g.subplots[0,0].set_title('r = '+str(r_fit)+' $\pm$ '+str(sigma_r_fit)+' , $A_d$ = '+str(Ad_fit)+' $\pm$ '+str(sigma_Ad_fit), fontsize=12)
            for ax in g.subplots[:,0]:
                ax.axvline(0.0, color='k', ls=':')
                ax.axvline(r_fit, color='gray', ls='--')
            # for ax in [g.subplots[1,0]]:
                # ax.axhline(Ad_fit, color='gray', ls='--')
            # for ax in [g.subplots[1,1]]:
                # ax.axvline(Ad_fit, color='gray', ls='--')

            pl.savefig(self.get_output('likelihood_on_r'))
            pl.close()

            likelihood_on_r_with_stat_and_sys_res( [r_fit, Ad_fit], make_figure=True, tag='_v2' )

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
                        pl.loglog( ell_v_loc, norm*Cl_dust_obs, label='estimated dust template @ 150GHz', linestyle='-', color='DarkGray', linewidth=2.0, alpha=0.8)
                        
                        # pl.loglog( ell_v_loc, norm*ClBB_obs, label='observed BB', color='red', linestyle='-', linewidth=2.0, alpha=0.8)
                        # pl.loglog( ell_v_loc, norm*Cov_model, label='modeled BB', color='k', linestyle='-', linewidth=2.0, alpha=0.8)
                        pl.loglog( ell_v_loc, norm*(ClBB_obs - Cl_noise[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]), 
                                                label='observed BB - actual noise = tot BB + residuals', 
                                                color='red', linestyle='-', linewidth=2.0, alpha=0.8)
                        # modeled noise-debiased BB spectrum
                        # which should correspond to primordial BB + lensing BB + foregrounds residuals
                        pl.loglog( ell_v_loc, norm*(Cov_model - Cl_cov_clean[1][(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])]), 
                                                    label='modeled BB - modeled noise = tot BB + residuals', 
                                                    color='k', linestyle='-', linewidth=2.0, alpha=0.8)
                        pl.loglog( ell_v_loc, norm*Cl_CMB_template_150GHz[(ell_v>=self.config['lmin'])&(ell_v<=self.config['lmax'])], linestyle=':', color='red', 
                                        linewidth=3.0, alpha=1.0, label='input CMB template @ 150GHz')
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
        if self.config['dust_marginalization']:
            to_be_saved = np.hstack((r_fit,  sigma_r_fit, Ad_fit, sigma_Ad_fit))
        else:
            to_be_saved = np.hstack((r_fit,  sigma_r_fit))
        np.savetxt(self.get_output('estimated_cosmo_params'), to_be_saved, comments=column_names)
        if ((not self.config['dust_marginalization']) and (not self.config['sync_marginalization'])):
            np.save(self.get_output('gridded_likelihood'), np.hstack((r_v,  gridded_likelihood)))
        else:
            np.save(self.get_output('gridded_likelihood'), samples)

if __name__ == '__main__':
    results = PipelineStage.main()

