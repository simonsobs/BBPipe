import numpy as np
import healpy as hp
import pymaster as nmt
from bbpipe import PipelineStage
from types_mine import FitsFile,TextFile,DummyFile,NpzFile
from param_manager import ParameterManager
import os

class BB_Likelihood(PipelineStage):
	name 			= 'BB_Likelihood'
	inputs			= [('bandpowers',TextFile),('bandpowers_noise',NpzFile)]
	outputs			= [('param_chains', NpzFile),('cmb_model',NpzFile)]
	#outputs			= []
	
	def read_bandpowers(self):
		f	= np.loadtxt(self.get_input('bandpowers'))
		# put the BB spectrum in bbdata
		# the BB spectrum is in column 4
		self.bbdata			= f[:,4]
		self.Nbandpowers	= len(f[:,0])
		# this is the maximum ell defined in the last bandpower 
		self.max_ell		= f[-1,1]
		self.bandpowers		= f[:,0:2]
		return
	def read_cov_matrix(self):
		f					= np.load(self.get_input('bandpowers_noise'))
		bandpowers_noise	= f['bandpowers_noise']
		# bandpowers_noise has a shape (Nsimul,7,Nbins)
		# we calculate the covariance matrix for BB
		cov_matrix_bb	= np.cov(np.transpose(bandpowers_noise[:,2,:]))
		self.invcov		= np.linalg.inv(cov_matrix_bb)
		return
	def load_cmb(self):
		#Loads the CMB BB spectrum as defined in the config file. 
		cmb_lensingfile = np.loadtxt(self.config['cmb_model']['cmb_templates'][0])
		cmb_bbfile = np.loadtxt(self.config['cmb_model']['cmb_templates'][1])

		self.cmb_ells = cmb_bbfile[:, 0]
		# this will cut the fiducial spectra to the maximum ell defined by the bins
		# get it from one of the ilc weights	
		mask = (self.cmb_ells <= self.max_ell) & (self.cmb_ells > 1)
		#print(len(mask))
		#print(len(cmb_lensingfile[:, 3][mask]))
		self.cmb_ells = self.cmb_ells[mask]

		nell = len(self.cmb_ells)
		self.cmb_tens = np.zeros([self.npol, nell])
		self.cmb_lens = np.zeros([self.npol, nell])
		self.cmb_scal = np.zeros([self.npol, nell])
		if 'B' in self.config['pol_channels']:
			ind = self.pol_order['B']
			self.cmb_tens[ind] = cmb_bbfile[:, 3][mask] #- cmb_lensingfile[:, 3][mask]
			self.cmb_lens[ind] = cmb_lensingfile[:, 3][mask]
		if 'E' in self.config['pol_channels']:
			ind = self.pol_order['E']
			self.cmb_tens[ind] = cmb_bbfile[:, 2][mask] - cmb_lensingfile[:, 2][mask]
			self.cmb_scal[ind] = cmb_lensingfile[:, 2][mask]
		np.savez(self.get_output('cmb_model'),cmb_tens=self.cmb_tens,cmb_lens=self.cmb_lens,cmb_scal=self.cmb_scal)
		return
		
	def setup_likelihood(self):
		# setup the bandpowers
		self.npol = len(self.config['pol_channels'])
		self.pol_order=dict(zip(self.config['pol_channels'],range(self.npol)))
		#Decide if you're using H&L
		self.use_handl = self.config['likelihood_type'] == 'h&l'
		
		self.read_bandpowers()
		self.read_cov_matrix()
		self.load_cmb()
		if self.use_handl:
			self.prepare_h_and_l()
		self.params = ParameterManager(self.config)
		return
	def minimizer(self):
        #Find maximum likelihood
		from scipy.optimize import minimize
		def chi2(par):
			c2=-2*self.lnprob(par)
			return c2
		res=minimize(chi2, self.params.p0, method="Powell")
		return res.x
	def prepare_h_and_l(self):
		fiducial_noise = self.bbfiducial + self.bbnoise
		self.Cfl_sqrt = np.array([sqrtm(f) for f in fiducial_noise])
		self.observed_cls = self.bbdata + self.bbnoise
		return 
	def lnprob(self, par):
		#Likelihood with priors. 
		prior = self.params.lnprior(par)
		if not np.isfinite(prior):
			return -np.inf

		params = self.params.build_params(par)
		if self.use_handl:
			dx = self.h_and_l_dx(params)
		else:
			dx = self.chi_sq_dx(params)
		like = -0.5 * np.einsum('i, ij, j',dx,self.invcov,dx)
		return prior + like
	def chi_sq_dx(self, params):
		"""
		Chi^2 likelihood. 
		"""
		model_dls = self.model(params)
		#return self.matrix_to_vector(self.bbdata - model_cls).flatten()
		return self.bbdata - model_dls
	
	def model(self, params):
		"""
		Defines the total model and integrates over the bandpasses and windows. 
		"""
		cmb_cell = (params['r_tensor'] * self.cmb_tens + params['A_lens'] * self.cmb_lens + self.cmb_scal)
		# in here i need to define a conversion factor if the fiducial input is in Dell.
		#self.bbdata is already in Dell

		#fg_scaling, rot_m = self.integrate_seds(params)  # [nfreq, ncomp], [ncomp,nfreq,[matrix]]
		#fg_cell = self.evaluate_power_spectra(params)  # [ncomp,ncomp,npol,npol,nell]

		# Add all components scaled in frequency (and HWP-rotated if needed)
		#cls_array_fg = np.zeros([self.nfreqs,self.nfreqs,self.n_ell,self.npol,self.npol])
		#fg_cell = np.transpose(fg_cell, axes = [0,1,4,2,3])  # [ncomp,ncomp,nell,npol,npol]
		cmb_cell = np.transpose(cmb_cell, axes = [1,0]) # [nell,npol,npol]
		"""
		for f1 in range(self.nfreqs):
			for f2 in range(f1,self.nfreqs):  # Note that we only need to fill in half of the frequencies
				cls=cmb_cell.copy()

				# Loop over component pairs
				for c1 in range(self.fg_model.n_components):
					mat1=rot_m[c1][f1]
					a1=fg_scaling[f1,c1]
					for c2 in range(self.fg_model.n_components):
						mat2=rot_m[c2][f2]
						a2=fg_scaling[f2,c2]
						# Rotate if needed
						clrot=rotate_cells_mat(mat2,mat1,fg_cell[c1,c2])
						# Scale in frequency and add
						cls += clrot*a1*a2
				cls_array_fg[f1,f2]=cls
		"""

		# Window convolution
		# I simplify this
		"""
		cls_array_list = np.zeros([self.n_bpws, self.nfreqs, self.npol, self.nfreqs, self.npol])
		for f1 in range(self.nfreqs):
			for p1 in range(self.npol):
				m1 = f1*self.npol+p1
				for f2 in range(f1,self.nfreqs):
					p0 = p1 if f1==f2 else 0
					for p2 in range(p0,self.npol):
						m2 = f2*self.npol+p2
						windows = self.windows[self.vector_indices[m1, m2]]
						clband = np.dot(windows, cls_array_fg[f1,f2,:,p1,p2])
						cls_array_list[:, f1, p1, f2, p2] = clband
						if m1!=m2:
							cls_array_list[:, f2, p2, f1, p1] = clband
		"""
		# this only works for BB !!!!!!
		# must be generalized
		model_bandpowers	= np.zeros((self.Nbandpowers))
		for n in range(self.Nbandpowers):
			mask = (self.cmb_ells >= self.bandpowers[n,0]) & (self.cmb_ells<self.bandpowers[n,1])
			indpol	= self.pol_order['B']
			model_bandpowers[n] = np.mean(cmb_cell[mask,indpol])
		return model_bandpowers
	def emcee_sampler(self):
		"""
		Sample the model with MCMC. 
		"""
		import emcee
		from multiprocessing import Pool
		fname_temp = self.get_output('param_chains')+'.h5'
		backend = emcee.backends.HDFBackend(fname_temp)
		nwalkers = self.config['nwalkers']
		n_iters = self.config['n_iters']
		ndim = len(self.params.p0)
		found_file = os.path.isfile(fname_temp)
		if not found_file:
			backend.reset(nwalkers,ndim)
			pos = [self.params.p0 + 1.e-3*np.random.randn(ndim) for i in range(nwalkers)]
			nsteps_use = n_iters
		else:
			print("Restarting from previous run")
			pos = None
			nsteps_use = max(n_iters-len(backend.get_chain()), 0)
		with Pool() as pool:
			sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,backend=backend)
			if nsteps_use > 0:
				sampler.run_mcmc(pos, nsteps_use, store=True, progress=True);
		return sampler

	def save_to_npz(self,fname,params,names,chi2):
		np.savez(fname,params=params,names=names,chi2=chi2)
		return
	
	def run(self):
		self.setup_likelihood()
		#print(self.invcov)
		if self.config.get('sampler')=='maximum_likelihood':
			sampler = self.minimizer()
			chi2 = -2*self.lnprob(sampler)
			self.save_to_npz(self.get_output('param_chains'),sampler,self.params.p_free_names,chi2)
			print("Best fit:")
			for n,p in zip(self.params.p_free_names,sampler):
				print(n+" = %.3lE" % p)
			print("Chi2: %.3lE" % chi2)
		if self.config.get('sampler')=='emcee':
			sampler = self.emcee_sampler()
			np.savez(self.get_output('param_chains'),chain=sampler.chain,names=self.params.p_free_names)
		print("Finished sampling")
		return
if __name__ == '__main__':
    cls = PipelineStage.main()
