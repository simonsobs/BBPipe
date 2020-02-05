import numpy as np
import emcee

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

fdir = '/mnt/zfsusers//mabitbol/BBPipe/updated_runs/combined_bandsangles/output2/'
reader = emcee.backends.HDFBackend(fdir+'sampler_out.h5')
x = np.load(fdir+'paramnames.npz')['arr_0']
labels = ['$%s$' %alllabels[k] for k in x]
#tau = reader.get_autocorr_time()
#burnin = int(5.*np.max(tau))
#thin = int(0.5*np.min(tau))
samples = reader.get_chain()
print(samples.shape)

samples = reader.get_chain(discard=5000, flat=True, thin=200) 
np.savez(fdir+'cleaned_chains', samples=samples, labels=labels)
