import numpy as np
import healpy as hp
import pymaster as nmt
from mpi4py import MPI
from bbpipe import PipelineStage
from types_mine import FitsFile,TextFile,DummyFile,NpzFile
from param_manager import ParameterManager
import os

class BB_Likelihood(PipelineStage):
	name 			= 'BB_Likelihood'
	inputs			= [('ell_bins_list',TextFile),('bandpowers_MC',NpzFile),('noise_bias',NpzFile),('NmtWorkspace',FitsFile)]
	outputs			= [('param_chains', NpzFile),('cmb_model',NpzFile)]
	
	def init_params(self):
		self.Nmc = self.config['Nsimul']
		self.bin_range = self.config['bin_range']
		self.method = self.config['method']
		self.nside 		= self.config['nside']
	
	def read_ell_bins(self,rank):
		bins_edges = np.loadtxt(self.get_input('ell_bins_list'))
		bins_edges_min = [] ; bins_edges_max = []
		for n in range(len(bins_edges)-1):
			bins_edges_min.append(int(bins_edges[n]))
			bins_edges_max.append(int(bins_edges[n+1]))
		self.bins = nmt.NmtBin.from_edges(bins_edges_min, bins_edges_max, is_Dell=True)
		self.Nbins_total = self.bins.get_n_bands() # this is the total number of bins, which is different from self.Nbandpowers, which is the number of bins that I will use for the likelihood
		self.max_ell = self.bins.get_ell_max(self.bin_range[1])
		self.Nbins_like = self.bin_range[1] - self.bin_range[0] + 1 # this is the number of bins I will use to do the likelihood Nbins_like <= Nbins_total
		if rank==0:
			print('The bins will be')
			for n in range(self.Nbins_total):
				print('bin %i, ell_min %i, ell_max %i'%(n,self.bins.get_ell_min(n),self.bins.get_ell_max(n)))
	def read_bandpowers(self,m,rank):
		# m is the iteration index, 1 is the BB spectrum, 
		self.bbdata			= self.bandpowers_MC[m,1,self.bin_range[0]:self.bin_range[1]+1]
		if rank==0:
			print('ell max is ',self.max_ell)
	
	def read_NmtWorkspace(self):
		self.workspace = nmt.NmtWorkspace()
		self.workspace.read_from(self.get_input('NmtWorkspace'))
	
	def setup_likelihood_global(self):
		# setup the bandpowers
		self.npol = len(self.config['pol_channels'])
		print('I will work with %i polarizations'%self.npol)
		self.pol_order=dict(zip(self.config['pol_channels'],range(self.npol)))
		#Decide if you're using H&L
		self.use_handl = self.config['likelihood_type'] == 'h&l'
		self.read_cov_matrix()
		self.load_cmb()
		self.couple_decouple_model_once()
	
	def read_cov_matrix(self):
		f					= np.load(self.get_input('bandpowers_MC'))
		self.bandpowers_MC	= f['bandpowers_MC']
		# we calculate the covariance matrix for BB
		# In bandpowers_MC 0 is EE, 1 is BB
		cov_matrix_bb	= np.cov(self.bandpowers_MC[:,1,self.bin_range[0]:self.bin_range[1]+1].T)
		self.invcov		= np.linalg.inv(cov_matrix_bb)
	def load_cmb(self):
		#Loads the CMB BB spectrum as defined in the config file. 
		# These have to be Cell, since when they are binned using bins.bin_cell they will be converted to Dell
		cmb_lensingfile = np.loadtxt(self.config['cmb_model']['cmb_templates'][0])
		cmb_bbfile = np.loadtxt(self.config['cmb_model']['cmb_templates'][1])
		self.cmb_ells = cmb_bbfile[:, 0]
		Nells = len(self.cmb_ells)
		self.cmb_tens = np.zeros((self.npol+2, Nells))
		self.cmb_lens = np.zeros((self.npol+2, Nells))
		self.cmb_scal = np.zeros((self.npol+2, Nells))
		if 'E' in self.config['pol_channels']:
			ind = 0
			self.cmb_lens[ind] = cmb_lensingfile[:, 2]
		if 'B' in self.config['pol_channels']:
			ind = 3
			self.cmb_tens[ind] = cmb_bbfile[:, 3] #- cmb_lensingfile[:, 3][mask]
			self.cmb_lens[ind] = cmb_lensingfile[:, 3]
		coupled_model = self.workspace.couple_cell( self.cmb_lens[:,0:3*self.nside] )
		decoupled_model = self.workspace.decouple_cell(coupled_model)
		# I take index=3 because we only want BB, and self.Nbins_like because we want the maximum bin for the likelihood
		self.Cfl_handl = decoupled_model[3,self.bin_range[0]:self.bin_range[1]+1]
		np.savez(self.get_output('cmb_model'),cmb_tens=self.cmb_tens,cmb_lens=self.cmb_lens,cmb_scal=self.cmb_scal)
	def couple_decouple_model_once(self):
		# since the coupling and decoupling with the matrix is linear, we do it only once for the lensing spectrum and once for the tensor spectrum.
		# We do this to avoid to have to couple/decouple the model spectrum every time we call the model inside the MCMC, which would take a long time
		indpol	= 3
		coupled_model_tens = self.workspace.couple_cell( self.cmb_tens[:,0:3*self.nside] )
		self.decoupled_model_tens = self.workspace.decouple_cell(coupled_model_tens)
		coupled_model_lens = self.workspace.couple_cell( self.cmb_lens[:,0:3*self.nside] )
		self.decoupled_model_lens = self.workspace.decouple_cell(coupled_model_lens)
	
	def setup_likelihood_iter(self):
		if self.use_handl:
			if self.method == 'hilc':
				# load the BB noise bias, which is index = 3 because is in the namaster order
				self.noise_bias = np.load(self.get_input('noise_bias'))['noise_bias'][3,:]
			elif self.method == 'nilc' or self.method == 'cnilc':
				# load the BB noise bias, which is index = 1 because we are working with two spin-0 field E B
				self.noise_bias = np.load(self.get_input('noise_bias'))['noise_bias'][1,self.bin_range[0]:self.bin_range[1]+1]
			self.prepare_h_and_l()
		self.params = ParameterManager(self.config)
	def minimizer(self):
        #Find maximum likelihood
		from scipy.optimize import minimize
		def chi2(par):
			c2=-2*self.lnprob(par)
			return c2
		res=minimize(chi2, self.params.p0, method="Powell")
		return res.x
	def prepare_h_and_l(self):
		fiducial_noise = self.Cfl_handl + self.noise_bias
		self.Cfl_sqrt = np.array([np.sqrt(f) for f in fiducial_noise])
		# bbnoise is the noise bias estimated in the previous step
		self.observed_cls = self.bbdata + self.noise_bias
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
	def h_and_l_dx(self, params):
		"""
		Hamimeche and Lewis likelihood. 
		Taken from Cobaya written by H, L and Torrado
		See: https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/_cmblikes_prototype/_cmblikes_prototype.py
		"""
		model_cls = self.model(params)
		dx_vec = np.zeros((model_cls.shape[0]))
		for k in range(model_cls.shape[0]):
			# Cl is the fiducial (that changes as the parameters change) plus the noise bias
			Cl = model_cls[k] + self.noise_bias[k]
			dx_vec[k] = self.h_and_l(Cl, self.observed_cls[k], self.Cfl_sqrt[k])
		return dx_vec
	def h_and_l(self, C, Chat, Cfl_sqrt):
		def g(x):
			return np.sign(x - 1) * np.sqrt(2 * np.maximum(0, x - np.log(x) - 1.0))
		inside = Chat/C # C^(-1/2)*C^(-1/2) = C^-1
		#print('inside',inside)
		Xg = Cfl_sqrt*g(inside)*Cfl_sqrt
		return Xg
	def chi_sq_dx(self, params):
		"""
		Chi^2 likelihood. 
		"""
		model_dls = self.model(params)
		#return self.matrix_to_vector(self.bbdata - model_cls).flatten()
		return self.bbdata - model_dls
	def couple_decouple_model_once(self):
		indpol	= 3
		coupled_model_tens = self.workspace.couple_cell( self.cmb_tens[:,0:3*self.nside] )
		self.decoupled_model_tens = self.workspace.decouple_cell(coupled_model_tens)
		coupled_model_lens = self.workspace.couple_cell( self.cmb_lens[:,0:3*self.nside] )
		self.decoupled_model_lens = self.workspace.decouple_cell(coupled_model_lens)
	def model(self, params):
		"""
		Defines the total model and integrates over the bandpasses and windows. 
		"""
		# We already binned the model tens and lens separately in couple_decouple_model_once(), we can do this because it is a linear operation
		#cmb_cell = (params['r_tensor'] * self.cmb_tens + params['A_lens'] * self.cmb_lens + self.cmb_scal)
		cmb_cell_binned = params['r_tensor'] * self.decoupled_model_tens + params['A_lens'] * self.decoupled_model_lens
		# in here i need to define a conversion factor if the fiducial input is in Dell.
		#self.bbdata is already in Dell
		#fg_scaling, rot_m = self.integrate_seds(params)  # [nfreq, ncomp], [ncomp,nfreq,[matrix]]
		#fg_cell = self.evaluate_power_spectra(params)  # [ncomp,ncomp,npol,npol,nell]
		# Add all components scaled in frequency (and HWP-rotated if needed)
		#cls_array_fg = np.zeros([self.nfreqs,self.nfreqs,self.n_ell,self.npol,self.npol])
		#fg_cell = np.transpose(fg_cell, axes = [0,1,4,2,3])  # [ncomp,ncomp,nell,npol,npol]
		# this only works for BB !!!!!!
		# must be generalized
		indpol	= 3
		model_bandpowers = cmb_cell_binned[indpol,self.bin_range[0]:self.bin_range[1]+1]
		return model_bandpowers
	def emcee_sampler(self,m):
		"""
		Sample the model with MCMC. 
		"""
		import emcee
		#from multiprocessing import Pool
		#fname_temp = self.get_output('param_chains')+'.h5.%04i'%m
		#backend = emcee.backends.HDFBackend(fname_temp)
		nwalkers = self.config['nwalkers']
		ndim = len(self.params.p0)
		#found_file = os.path.isfile(fname_temp)
		pos = [self.params.p0 + 1.e-3*np.random.randn(ndim) for i in range(nwalkers)]
		nsteps_use = self.config['n_iters']
		#if not found_file:
			#backend.reset(nwalkers,ndim)
		#	pos = [self.params.p0 + 1.e-3*np.random.randn(ndim) for i in range(nwalkers)]
		#	nsteps_use = n_iters
		#else:
		#	print('Restarting from previous run')
		#	pos = None
			#nsteps_use = max(n_iters-len(backend.get_chain()), 0)
		#with Pool() as pool:
			#sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,backend=backend)
		sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
		if nsteps_use > 0:
			sampler.run_mcmc(pos, nsteps_use, store=True, progress=True);
		return sampler
	def save_to_npz(self,fname,params,names,chi2):
		np.savez(fname,params=params,names=names,chi2=chi2)
	def run(self):
		# setup the mpi communicator
		comm = MPI.COMM_WORLD
		size_pool = comm.Get_size()
		rank = comm.Get_rank()
		self.init_params()
		self.read_ell_bins(rank)
		self.read_NmtWorkspace()
		self.setup_likelihood_global()
		for m in range(rank*self.Nmc//size_pool,(rank+1)*self.Nmc//size_pool):
			print('Starting with iter=%04i'%m)
			self.read_bandpowers(m,rank)
			self.setup_likelihood_iter()
			if self.config.get('sampler')=='maximum_likelihood':
				sampler = self.minimizer()
				chi2 = -2*self.lnprob(sampler)
				self.save_to_npz(self.get_output('param_chains'),sampler,self.params.p_free_names,chi2)
				print('Best fit:')
				for n,p in zip(self.params.p_free_names,sampler):
					print(n+' = %.3lE' % p)
				print('Chi2: %.3lE' % chi2)
			if self.config.get('sampler')=='emcee':
				sampler = self.emcee_sampler(m)
				np.savez(self.get_output('param_chains')+'.%04i'%m,chain=sampler.chain,names=self.params.p_free_names)
			print('Finished sampling for iter=%04i'%m)
		return
if __name__ == '__main__':
    cls = PipelineStage.main()
