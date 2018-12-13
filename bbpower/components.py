import numpy as np
# constants
TCMB = 2.725  # Kelvin
hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS
clight = 299792458.0  # MKS

def normed_cmb_thermo_units(nu):
    X = hplanck * nu / (kboltz * TCMB)
    eX = np.exp(X)
    return eX * X**4 / (eX - 1.)**2 

def normed_plaw(ell, alpha):
    ell0 = 80.
    return (ell/ell0)**alpha 

def normed_synch(nu, beta):
    nu0 = 30.e9
    return (nu/nu0)**(2.+beta)

def normed_dust(nu, beta):
    Td = 19.6 # K
    nu0 = 353.e9
    X = hplanck * nu / (kboltz * Td)
    X0 = hplanck * nu0 / (kboltz * Td)
    return (nu/nu0)**(3.+beta) * (np.exp(X0) - 1.) / (np.exp(X) - 1.)


# don't need these 
def blackbody(nu, T):
    X = hplanck * nu / (kboltz * T)
    return 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(X) - 1.0))

def dbbdt(nu, A):
    X = hplanck * nu / (kboltz * TCMB)
    gf = (2.0 * hplanck**2 * nu**4) / (kboltz * clight**2) * (1./TCMB**2) * np.exp(X) / (np.exp(X) - 1.0)**2
    return A * gf
