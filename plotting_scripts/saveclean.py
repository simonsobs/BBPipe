import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
from getdist import plots, MCSamples, loadMCSamples
import emcee
from fsettings import alllabels

def check_chains(fdir, savename):
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
        burnin = 1500
        thin = 20
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin) 
    np.savez(fdir+'cleaned_chains', samples=samples, names=x['names'], labels=labels, p0=x['p0'])
    return

fdir = '/mnt/zfsusers/mabitbol/BBPipe/final_runs/dphi/sinuous0/'
sname = 'dphi_sinuous_noangle'
check_chains(fdir, sname)

fdir = '/mnt/zfsusers/mabitbol/BBPipe/final_runs/all/final/'
sname = 'all'
check_chains(fdir, sname)

fdir = '/mnt/zfsusers/mabitbol/BBPipe/final_runs/all/finaleb/'
sname = 'all_eb'
check_chains(fdir, sname)

if False:
    fdir = '/mnt/zfsusers/mabitbol/BBPipe/final_runs/bandpass/r01/'
    sname = 'bandpass_r0.01'
    check_chains(fdir, sname)

    fdir = '/mnt/zfsusers/mabitbol/BBPipe/final_runs/bandpass/r01decorr/'
    sname = 'bandpass_r0.01_decorr'
    check_chains(fdir, sname)

    fdir = '/mnt/zfsusers/mabitbol/BBPipe/final_runs/dphi/r01/'
    sname = 'dphi_r0.01'
    check_chains(fdir, sname)

    fdir = '/mnt/zfsusers/mabitbol/BBPipe/final_runs/dphi/r01eb/'
    sname = 'dphi_r0.01_eb'
    check_chains(fdir, sname)

    fdir = '/mnt/zfsusers/mabitbol/BBPipe/final_runs/dphi/sinuous/'
    sname = 'dphi_sinuous'
    check_chains(fdir, sname)


    fdir = '/mnt/zfsusers/mabitbol/BBPipe/final_runs/dphi/sinuousdphi1/'
    sname = 'dphi_sinuous_dphi1'
    check_chains(fdir, sname)

