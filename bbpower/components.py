import numpy as np
# constants
TCMB = 2.725  # Kelvin
hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS
clight = 299792458.0  # MKS
# consider using numba or cython...or really F90 or C++

def blackbody(nu, T):
    X = hplanck * nu / (kboltz * T)
    return 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(X) - 1.0))

def dbbdt(nu, A):
    X = hplanck * nu / (kboltz * TCMB)
    gf = (2.0 * hplanck**2 * nu**4) / (kboltz * clight**2) * (1./TCMB**2) * np.exp(X) / (np.exp(X) - 1.0)**2
    return A * gf

def normed_plaw(ell, alpha):
    ell0 = 80.
    return (ell/ell0)**alpha 

def normed_synch(nu, beta):
    nu0 = 30.e9
    return (nu/nu0)**beta

def synch(nu, beta):
    nu0 = 30.e9
    units = blackbody(nu0, TCMB) / blackbody(nu, TCMB)
    return (nu/nu0)**beta * units

def normed_dust(nu, beta):
    # biceps model for mbb (I think)
    Td = 19.6 # K
    nu0 = 353.e9
    return (nu/nu0)**(beta-2.) * blackbody(nu, Td) / blackbody(nu0, Td)
    
def dust(nu, beta):
    # biceps model for mbb (I think)
    Td = 19.6 # K
    nu0 = 353.e9
    units = blackbody(nu0, TCMB) / blackbody(nu, TCMB)
    return (nu/nu0)**(beta-2.) * blackbody(nu, Td) / blackbody(nu0, Td) * units

