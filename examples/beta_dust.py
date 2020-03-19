import pysm
from pysm.nominal import models

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt


# Map parameters                                                                                                                                                       
nside = 512 #int(len(ells)/3)
lmax = 3*nside-1
ells = np.arange(lmax+1)


'''
Power law model 
'''

# Parameters
#---------------------------------------------------

# A_beta=1e-6

beta_sync_0=-3.

# Gamma must be less than -2 for convergence in 0x2 term
gamma_beta = -2.5 

# Critical value (for convergence) corresponds to 3 sigma for the std map
# SO freqs: 27., 39., 93., 145., 225., 280. GHz
# ~10 is the ratio between our highest and lowest frequency 280/27 GHz
sigma_crit = 2/np.log(10) 

# model the standard deviation map as a function of gamma
# std = a * (-gamma)^b * exp(c * gamma)
# best fit parameters a b c
a = 4.16190627
b = -3.28619789
c = -2.56282892
sigma_emp = a * (-gamma_beta)**b * np.exp(c*gamma_beta)                       

A_beta = (sigma_crit / sigma_emp)**2


print(A_beta)


# Calculate power spectrum (input)
#---------------------------------------------------

#(Gaussian, power law: C_{\ell} = A \cdot (\frac{\ell+0.001}{80.})^\alpha $) #beta_cls = sigma_beta * np.ones_like(ells)                                                                                                                                       
#dlfac=2*np.pi/(ells*(ells+1.)); dlfac[0]=1
dlfac=1
dl_betaSync= A_beta * ((ells+0.001) / 80.)**gamma_beta #
cl_betaSync = dl_betaSync * dlfac 


plt.figure()
plt.semilogy(ells, cl_betaSync, 'b',label="input power spectrum")
plt.xlabel('$\ell$')
plt.ylabel('$\ell(\ell+1) C_\ell/2\pi$')
plt.legend()
plt.grid()
plt.show()


# Map from power spectrum 
#---------------------------------------------------

map_beta = hp.synfast(cl_betaSync, nside, new=True, verbose=False)
delta_beta = map_beta *  sigma_crit / sigma_emp   #delta_beta
                                                 #Gaussian random field
                                                 #beta_map -= np.mean(beta_map) + 3.2
#map_beta -= (np.mean(map_beta) - delta_beta)
#hp.mollview(map_beta, title='Scaling Index beta_sync') #unit='$\\beta$'


map_beta -= np.mean(map_beta) -  delta_beta
#map_beta -= -3  -  delta_beta


'''
print('delta_beta 3.2')
print(delta_beta)
print('mean(map_beta) -3')
print(np.mean(map_beta))
'''
hp.mollview(map_beta,  title='Scaling Index beta_sync') #unit='$\\beta$'
hp.mollview(map_beta,  title='Scaling Index beta_sync') #unit='$\\beta$'

plt.show() 


# Power spectrum from map (output)
#---------------------------------------------------

cl_betaSync_out = hp.anafast(map_beta, lmax=lmax)  
dl_betaSync_out = cl_betaSync_out / dlfac


# Compare input and output power spectra 
#---------------------------------------------------

plt.figure()
plt.semilogy(ells, dl_betaSync, 'b',label="input power spectrum")
plt.semilogy(ells, dl_betaSync_out, 'r',label="output power spectrum")
plt.xlabel('$\ell$')
plt.ylabel('$\ell(\ell+1) C_\ell/2\pi$')
plt.legend()
plt.grid()
plt.show()


