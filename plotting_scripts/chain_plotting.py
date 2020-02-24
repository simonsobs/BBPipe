import numpy as np
import emcee
import corner
from getdist import plots, MCSamples, loadMCSamples
from IPython.display import display, Math
import matplotlib.pyplot as plt

alllabels = {'A_lens':'A_{lens}', 'r_tensor':'r', 'beta_d':'\\beta_d', 'epsilon_ds':'\epsilon', 
             'alpha_d_ee':'\\alpha^{EE}_d', 'amp_d_ee':'A^{EE}_d', 'alpha_d_bb':'\\alpha^{BB}_d', 
             'amp_d_bb':'A^{BB}_d', 'beta_s':'\\beta_s', 'alpha_s_ee':'\\alpha^{EE}_s', 
             'amp_s_ee':'A^{EE}_s', 'alpha_s_bb':'\\alpha^{BB}_s', 'amp_s_bb':'A^{BB}_s', 
             'amp_d_eb':'A^{EB}_d', 'alpha_d_eb':'\\alpha^{EB}_d', 'amp_s_eb':'A^{EB}_s', 
             'alpha_s_eb':'\\alpha^{EB}_s',
             'gain_1':'G_1', 'gain_2':'G_2', 'gain_3':'G_3', 'gain_4':'G_4', 'gain_5':'G_5', 'gain_6':'G_6',
             'shift_1':'\Delta\\nu_1', 'shift_2':'\Delta\\nu_2', 'shift_3':'\Delta\\nu_3', 
             'shift_4':'\Delta\\nu_4', 'shift_5':'\Delta\\nu_5', 'shift_6':'\Delta\\nu_6', 
             'angle_1':'\phi_1', 'angle_2':'\phi_2', 'angle_3':'\phi_3', 'angle_4':'\phi_4', 
             'angle_5':'\phi_5', 'angle_6':'\phi_6', 'dphi1_1':'\Delta\phi_1', 'dphi1_2':'\Delta\phi_2', 
             'dphi1_3':'\Delta\phi_3', 'dphi1_4':'\Delta\phi_4', 'dphi1_5':'\Delta\phi_5', 'dphi1_6':'\Delta\phi_6', 
             'decorr_amp_d':'\Delta_d', 'decorr_amp_s':'\Delta_s'}

def save_cleaned_chains(fdir, savename):
    reader = emcee.backends.HDFBackend(fdir+'params_out.npz.h5')
    x = np.load(fdir+'params_out.npz')
    labels = ['$%s$' %alllabels[k] for k in x['names']]

    try:
        tau = reader.get_autocorr_time()
        burnin = int(10.*np.max(tau))
        thin = int(0.5*np.min(tau))
        print(burnin, thin)
    except Exception as e:
        print(e)
        tau = reader.get_autocorr_time(tol=0)
        print(tau)
        burnin = 2000
        thin = 40

    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    np.savez(fdir+'cleaned_chains', samples=samples, names=x['names'], labels=labels, p0=x['p0'])
    return

def plot_walkers(fdir, savename):
    reader = emcee.backends.HDFBackend(fdir+'params_out.npz.h5')
    samples = reader.get_chain()
    x = np.load(fdir+'params_out.npz')
    labels = ['$%s$' %alllabels[k] for k in x['names']]
 
    ndim = samples.shape[-1]
    fig, axes = plt.subplots(ndim, figsize=(16, 20), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[::20, :, i], "k", alpha=0.25)
        ax.set_ylabel(labels[i])
    savefig('/mnt/zfsusers/mabitbol/notebooks/data_and_figures/'+savename+'.png', fmt='png')
    return

## These require you to run save_cleaned_chains first! ##
def corner_triangle(fdir, savename, bins=100):
    samps = np.load(fdir+'cleaned_chains.npz')
    samples = samps['samples']
    ndim = samples.shape[-1]
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], samps['labels'][i].strip('$'))
        display(Math(txt))
    
    fig = corner.corner(samples, labels=samps['labels'], truths=samps['p0'], bins=bins, smooth=1., 
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 16});
    savefig('/mnt/zfsusers/mabitbol/notebooks/data_and_figures/corner_'+savename+'.png', fmt='png')
    return 

def getdist_clean(fdir):
    sampler = np.load(fdir+'cleaned_chains.npz')
    gd_samps = MCSamples(samples=sampler['samples'], names=sampler['labels'])
    return gd_samps

def gdplot(fdir, savename):
    samps = getdist_clean(fdir)
    print(samps.getTable().tableTex())
    
    g = plots.getSubplotPlotter()
    g.triangle_plot([samps], shaded=True)
    savefig('/mnt/zfsusers/mabitbol/notebooks/data_and_figures/'+savename+'_triangle.png', fmt='png')
    close()
    
    g = plots.getSinglePlotter(width_inch=12)
    g.settings.axes_fontsize = 16
    g.plot_1d([samps], '$r$')
    grid()
    savefig('/mnt/zfsusers/mabitbol/notebooks/data_and_figures/'+savename+'_r.png', fmt='png')
    close()

