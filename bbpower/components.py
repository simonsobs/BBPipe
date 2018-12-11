import numpy as np
# constants
TCMB = 2.725  # Kelvin
hplanck = 6.626068e-34  # MKS
kboltz = 1.3806503e-23  # MKS
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

def normed_dust(nu, beta):
    # biceps model for mbb (I think)
    Td = 19.6 # K
    nu0 = 353.e9
    return (nu/nu0)**(Bd-2.) * blackbody(nu, Td) / blackbody(nu0, Td)
    

def model(params):
    # offensively bad model here, but it is just to test our understanding of the data and usage of sacc. 
    # need to implement CMB / figure out how to do unit conversions. 
    r, A_s, A_d, beta_s, beta_d, alpha_s, alpha_d, epsilon = params
    
    # load bmode spectrum
    # cmb_bb = np.load(bmodes)
    
    # precompute SEDs
    empty = np.empty(len(tns)) # number of datasets = 12
    seds = {'synch':[], 'dust': [], 'cmb':[]}
    for tn in tns:
        # integrate bandpasses 
        nus = bpasses[tn][0]
        bpass = bpasses[tn][1]
        nom_synch = fgs.normed_synch(nus, beta_s)
        nom_dust = fgs.normed_dust(nus, beta_d)
        seds['synch'][tn] = np.dot(nom_synch, bpass)
        seds['dust'][tn] = np.dot(nom_dust, bpass)
        # cmb
    # seds have shape 12
    
    # precompute power laws in ell 
    nom_synch_spectrum = fgs.normed_plaw(bpw_l, alpha_s)
    nom_dust_spectrum = fgs.normed_plaw(bpw_l, alpha_d)
    nom_cross_spectrum = np.sqrt(nom_synch_spectrum * nom_dust_spectrum)
    # cmb
    # these have length 600
    
    cls_array_list = [] 
    for t1,t2,typ,ells,ndx in order:
        # questionable lmao 
        if typ == b'BB':
            # multiply and sum (and get the right order and such)
            windows = s.binning.windows[ndx]
            synch_spectrum = [np.dot(w, nom_synch_spectrum) for w in windows]
            dust_spectrum = [np.dot(w, nom_dust_spectrum) for w in windows]
            cross_spectrum = [np.dot(w, nom_cross_spectrum) for w in windows]        
            #cmb 
            # these have length 9 
            
            fs1 = seds['synch'][t1]
            fs2 = seds['synch'][t2]
            fd1 = seds['dust'][t1]
            fd2 = seds['dust'][t2]
            
            synch = A_s * fs1*fs2 * synch_spectrum
            dust = A_d * fd1*fd2 * dust_spectrum
            cross = np.sqrt(A_s * A_d) * (fs1*fd2 + fs2*fd1) * cross_spectrum
            #cmb 
            
            model = cmb + synch + dust + cross
            
            cls_array_list.append(model)
    
    return cls_array_list.reshape(len(indx), ) 



