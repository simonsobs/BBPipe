import numpy as np
import matplotlib.pyplot as plt

"""
 keys
 ['params', 'names', 'chi2']
 names
 ['A_lens', 'r_tensor', 'epsilon_ds', 'alpha_d_bb', 'amp_d_bb', 'alpha_s_bb', 'amp_s_bb']

"""

def import_r(simulation):
    """ Returns the r fit value for a given simulation"""
    a = np.load(simulation+'/param_chains.npz')
    return a['params'][1]

def rs_list(sims_list):
    """ Returns a new list which contains the r fit values for 
    a given list of simulations"""
    r_list = []
    for sim in sims_list:
        new_elem = import_r(sim)
        r_list.append(new_elem)
    return r_list

def import_chi2(simulation):
    """ Returns the chi^2 value for a given simulation"""
    a = np.load(simulation+'/param_chains.npz')
    return a['chi2']

def chi2_list(sims_list):
    """ Returns a new list which contains the chi^2 values for 
    a given list of simulations"""
    chi_sq_list = []
    for sim in sims_list:
        new_elem = import_chi2(sim)
        chi_sq_list.append(new_elem)
    return chi_sq_list

def import_amp_d_beta(simulation):
    a = np.load(simulation+'/param_chains.npz')
    return a['params'][6]
    
def amp_d_beta_list(sims_list):
    amp_d_beta_list = []
    for sim in sims_list:
        new_elem = import_amp_d_beta(sim)
        amp_d_beta_list.append(new_elem)
    return amp_d_beta_list

def import_gamma_d_beta(simulation):
    a = np.load(simulation+'/param_chains.npz')
    return a['params'][7]
    
def gamma_d_beta_list(sims_list):
    gamma_d_beta_list = []
    for sim in sims_list:
        new_elem = import_gamma_d_beta(sim)
        gamma_d_beta_list.append(new_elem)
    return gamma_d_beta_list

def import_beta_d(simulation):
    a = np.load(simulation+'/param_chains.npz')
    return a['params'][2]
    
def beta_d_list(sims_list):
    beta_d_list = []
    for sim in sims_list:
        new_elem = import_beta_d(sim)
        beta_d_list.append(new_elem)
    return beta_d_list

seed = np.arange(1300, 1305, 1)
sigma_dust = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
directory = "/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/"

# Sims seed 1300, gaussian beta dust
prefix_1300 = directory+'test_new_simulations_%d/new_sim_ns256_seed%d_pysm'%(seed[0], seed[0])
allsims_1300 = [prefix_1300+'_sigD0sigS0',
                prefix_1300+'_sigD1sigS0',
                prefix_1300+'_sigD2sigS0',
                prefix_1300+'_sigD3sigS0',
                prefix_1300+'_sigD4sigS0',
                prefix_1300+'_sigD5sigS0',
                prefix_1300+'_sigD6sigS0',
                prefix_1300+'_sigD7sigS0',
                prefix_1300+'_sigD8sigS0',
                prefix_1300+'_sigD9sigS0',
                prefix_1300+'_sigD10sigS0',
                prefix_1300+'_sigD11sigS0',
                prefix_1300+'_sigD12sigS0',
                prefix_1300+'_sigD13sigS0',
                prefix_1300+'_sigD14sigS0',
                prefix_1300+'_sigD15sigS0',
                prefix_1300+'_sigD20sigS0']

# Sims seed 1301, gaussian beta dust
prefix_1301 = directory+'test_new_simulations_%d/new_sim_ns256_seed%d_pysm'%(seed[1], seed[1])
allsims_1301 = [prefix_1301+'_sigD0sigS0',
                prefix_1301+'_sigD1sigS0',
                prefix_1301+'_sigD2sigS0',
                prefix_1301+'_sigD3sigS0',
                prefix_1301+'_sigD4sigS0',
                prefix_1301+'_sigD5sigS0',
                prefix_1301+'_sigD6sigS0',
                prefix_1301+'_sigD7sigS0',
                prefix_1301+'_sigD8sigS0',
                prefix_1301+'_sigD9sigS0',
                prefix_1301+'_sigD10sigS0',
                prefix_1301+'_sigD11sigS0',
                prefix_1301+'_sigD12sigS0',
                prefix_1301+'_sigD13sigS0',
                prefix_1301+'_sigD14sigS0',
                prefix_1301+'_sigD15sigS0',
                prefix_1301+'_sigD20sigS0']

# Sims seed 1302, gaussian beta dust
prefix_1302 = directory+'test_new_simulations_%d/new_sim_ns256_seed%d_pysm'%(seed[2], seed[2])
allsims_1302 = [prefix_1302+'_sigD0sigS0',
                prefix_1302+'_sigD1sigS0',
                prefix_1302+'_sigD2sigS0',
                prefix_1302+'_sigD3sigS0',
                prefix_1302+'_sigD4sigS0',
                prefix_1302+'_sigD5sigS0',
                prefix_1302+'_sigD6sigS0',
                prefix_1302+'_sigD7sigS0',
                prefix_1302+'_sigD8sigS0',
                prefix_1302+'_sigD9sigS0',
                prefix_1302+'_sigD10sigS0',
                prefix_1302+'_sigD11sigS0',
                prefix_1302+'_sigD12sigS0',
                prefix_1302+'_sigD13sigS0',
                prefix_1302+'_sigD14sigS0',
                prefix_1302+'_sigD15sigS0',
                prefix_1302+'_sigD20sigS0']

# Sims seed 1303, gaussian beta dust
prefix_1303 = directory+'test_new_simulations_%d/new_sim_ns256_seed%d_pysm'%(seed[3], seed[3])
allsims_1303 = [prefix_1303+'_sigD0sigS0',
                prefix_1303+'_sigD1sigS0',
                prefix_1303+'_sigD2sigS0',
                prefix_1303+'_sigD3sigS0',
                prefix_1303+'_sigD4sigS0',
                prefix_1303+'_sigD5sigS0',
                prefix_1303+'_sigD6sigS0',
                prefix_1303+'_sigD7sigS0',
                prefix_1303+'_sigD8sigS0',
                prefix_1303+'_sigD9sigS0',
                prefix_1303+'_sigD10sigS0',
                prefix_1303+'_sigD11sigS0',
                prefix_1303+'_sigD12sigS0',
                prefix_1303+'_sigD13sigS0',
                prefix_1303+'_sigD14sigS0',
                prefix_1303+'_sigD15sigS0',
                prefix_1303+'_sigD20sigS0']

# Sims seed 1304, gaussian beta dust
prefix_1304 = directory+'test_new_simulations_%d/new_sim_ns256_seed%d_pysm'%(seed[4], seed[4])
allsims_1304 = [prefix_1304+'_sigD0sigS0',
                prefix_1304+'_sigD1sigS0',
                prefix_1304+'_sigD2sigS0',
                prefix_1304+'_sigD3sigS0',
                prefix_1304+'_sigD4sigS0',
                prefix_1304+'_sigD5sigS0',
                prefix_1304+'_sigD6sigS0',
                prefix_1304+'_sigD7sigS0',
                prefix_1304+'_sigD8sigS0',
                prefix_1304+'_sigD9sigS0',
                prefix_1304+'_sigD10sigS0',
                prefix_1304+'_sigD11sigS0',
                prefix_1304+'_sigD12sigS0',
                prefix_1304+'_sigD13sigS0',
                prefix_1304+'_sigD14sigS0',
                prefix_1304+'_sigD15sigS0',
                prefix_1304+'_sigD20sigS0']

r_fit_1300 = rs_list(allsims_1300)
r_fit_1301 = rs_list(allsims_1301)
r_fit_1302 = rs_list(allsims_1302)
r_fit_1303 = rs_list(allsims_1303)
r_fit_1304 = rs_list(allsims_1304)

plt.figure()
plt.scatter(sigma_dust, r_fit_1300, c='k', label='sims 1')
plt.scatter(sigma_dust, r_fit_1301, c='b', label='sims 2')
plt.scatter(sigma_dust, r_fit_1302, c='m', label='sims 3')
plt.scatter(sigma_dust, r_fit_1303, c='c', label='sims 4')
plt.scatter(sigma_dust, r_fit_1304, c='r', label='sims 5')
plt.legend()
plt.title('r, Mom=False')
plt.ylabel(r"r fit", fontsize = 10)
plt.xlabel(r"amplitude variation", fontsize = 10)
plt.savefig('r_fit_momF_all_neww.png')

chi_sq_1300 = chi2_list(allsims_1300)
n_dof = 56.7 * np.ones_like(chi_sq_1300)
chi_sq_1300 = chi2_list(allsims_1300) / n_dof
chi_sq_1301 = chi2_list(allsims_1301) / n_dof
chi_sq_1302 = chi2_list(allsims_1302) / n_dof
chi_sq_1303 = chi2_list(allsims_1303) / n_dof
chi_sq_1304 = chi2_list(allsims_1304) / n_dof

plt.figure()
plt.scatter(sigma_dust, chi_sq_1300, c='k', label='Sims 1300')
plt.scatter(sigma_dust, chi_sq_1301, c='b', label='Sims 1301')
plt.scatter(sigma_dust, chi_sq_1302, c='m', label='Sims 1302')
plt.scatter(sigma_dust, chi_sq_1303, c='c', label='Sims 1303')
plt.scatter(sigma_dust, chi_sq_1304, c='r', label='Sims 1304')
plt.legend()
plt.title('Chi^2, Mom=False')
plt.ylabel(r"Chi^2", fontsize = 10)
plt.xlabel(r"amplitude variation", fontsize = 10)
plt.savefig('chi_sq_momF_all_neww.png')

amp_d_beta_1300 = amp_d_beta_list(allsims_1300)
amp_d_beta_1301 = amp_d_beta_list(allsims_1301)
amp_d_beta_1302 = amp_d_beta_list(allsims_1302)
amp_d_beta_1303 = amp_d_beta_list(allsims_1303)
amp_d_beta_1304 = amp_d_beta_list(allsims_1304)

plt.figure()
plt.scatter(sigma_dust, amp_d_beta_1300, c='k', label='Sim 1')
plt.scatter(sigma_dust, amp_d_beta_1301, c='b', label='Sim 2')
plt.scatter(sigma_dust, amp_d_beta_1302, c='m', label='Sim 3')
plt.scatter(sigma_dust, amp_d_beta_1303, c='c', label='Sim 4')
plt.scatter(sigma_dust, amp_d_beta_1304, c='r', label='Sim 5')
plt.xlim(0., 20.)
plt.legend()
plt.title('Amplitude beta dust, Mom=True')
plt.ylabel(r"amplitude dust beta", fontsize = 10)
plt.xlabel(r"amplitude variation", fontsize = 10)
plt.savefig('amp_d_beta_momT_closer_neww.png')

gamma_d_beta_1300 = gamma_d_beta_list(allsims_1300)
gamma_d_beta_1301 = gamma_d_beta_list(allsims_1301)
gamma_d_beta_1302 = gamma_d_beta_list(allsims_1302)
gamma_d_beta_1303 = gamma_d_beta_list(allsims_1303)
gamma_d_beta_1304 = gamma_d_beta_list(allsims_1304)

plt.figure()
plt.scatter(sigma_dust, gamma_d_beta_1300, c='k', label='Sim 1')
plt.scatter(sigma_dust, gamma_d_beta_1301, c='b', label='Sim 2')
plt.scatter(sigma_dust, gamma_d_beta_1302, c='m', label='Sim 3')
plt.scatter(sigma_dust, gamma_d_beta_1303, c='c', label='Sim 4')
plt.scatter(sigma_dust, gamma_d_beta_1304, c='r', label='Sim 5')
plt.xlim(0., 20.)
plt.legend()
plt.title('Gamma dust, Mom=True')
plt.ylabel(r"gamma dust beta", fontsize = 10)
plt.xlabel(r"amplitude variation", fontsize = 10)
plt.savefig('gamma_d_beta_momT_closer_neww.png')

beta_d_1300 = beta_d_list(allsims_1300)
beta_d_1301 = beta_d_list(allsims_1301)
beta_d_1302 = beta_d_list(allsims_1302)
beta_d_1303 = beta_d_list(allsims_1303)
beta_d_1304 = beta_d_list(allsims_1304)

plt.figure()
plt.scatter(sigma_dust, beta_d_1300, c='k', label='Sim 1')
plt.scatter(sigma_dust, beta_d_1301, c='b', label='Sim 2')
plt.scatter(sigma_dust, beta_d_1302, c='m', label='Sim 3')
plt.scatter(sigma_dust, beta_d_1303, c='c', label='Sim 4')
plt.scatter(sigma_dust, beta_d_1304, c='r', label='Sim 5')
plt.xlim(0., 20.)
plt.ylim(1.5, 1.7)
plt.legend()
plt.title('Beta dust, Mom=True')
plt.ylabel(r"Beta dust", fontsize = 10)
plt.xlabel(r"amplitude variation", fontsize = 10)
plt.savefig('beta_d_momT_closer_neww.png')
