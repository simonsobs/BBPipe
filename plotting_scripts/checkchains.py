import numpy as np
import emcee
from fsettings import alllabels

def check_chains(fdir, savename):
    reader = emcee.backends.HDFBackend(fdir+'params_out.npz.h5')
    x = np.load(fdir+'params_out.npz')
    labels = ['$%s$' %alllabels[k] for k in x['names']]
    
    try:
        tau = reader.get_autocorr_time(tol=10)
        burnin = int(10.*np.max(tau))
        thin = int(0.5*np.min(tau))
        print(burnin, thin)
    except Exception as e:
        print(e)
        burnin = 2000
        thin = 40

    samples = reader.get_chain(discard=burnin, flat=True, thin=thin) 
    np.savez(fdir+'cleaned_chains', samples=samples, names=x['names'], labels=labels, p0=x['p0'])
    return

fdir = '/mnt/zfsusers/mabitbol/BBPipe/final_runs/bandpass/r01/'
sname = 'bandpass_r01'
check_chains(fdir, sname)
