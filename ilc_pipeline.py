import numpy as np
import healpy as hp
from mpi4py import MPI
import pymaster as nmt
from bbpipe import PipelineStage
from types_mine import FitsFile,TextFile,DummyFile,NpzFile,PklFile
from PixelILC import doNILC_SHTSmoothing_SingleField, doCNILC_ThermalDust_SHTSmoothing_SingleField
import pickle
import gc

class BB_HILC(PipelineStage):
	name 			= 'BB_HILC'
	inputs			= [('ell_bins_list',TextFile),('freqs_list',TextFile),('masks',TextFile),('freq_maps_list_MC',TextFile),('needlet_windows',TextFile),('fwhms_dominion',TextFile),('nside_per_window',TextFile),('which_channels_which_bands',TextFile),('beams_list',TextFile),('beta_dust_map',FitsFile),('T_dust_map',FitsFile)]
	
	outputs			= [('cl_cmb',FitsFile),('ilc_weights',NpzFile),('ilc_weights_multi',PklFile),('covariance_pixel',PklFile),('bands',NpzFile),('cl_cmb_MC',NpzFile),('ilc_weights_MC',NpzFile),('reconstruced_cmb_MC',NpzFile),('NILC_weights',NpzFile)]

	def init_params(self):
		self.nside 		= self.config['nside']
		self.npix 		= hp.nside2npix(self.nside)
		self.verbose	= self.config['verbose']
		self.method = self.config['method']
		self.Nmc = self.config['Nsimul']
		self.beam_band = int(self.config['beam_band'])
		self.Nthreads = int(self.config['Nthreads'])
		
		if self.method == 'hilc':
			self.weights_option = self.config['weights_option']
		if self.method == 'nilc' or self.method == 'cnilc':
			# number of nilc windows
			self.Nw = self.config['Nwindows']
			# maximum multipole defined in the nilc windows
			self.l_max = self.config['ell_max']
			self.weights_mode = self.config['weights_mode']
	def read_freqs(self):
		# read freq value and count the number of bands
		freqs	= []
		f			= open(self.get_input('freqs_list'),'r')
		self.nbands	= 0
		for line in f:
			freqs.append(float(line.strip()))
			self.nbands += 1
		# transform to numpy array
		self.freqs	= np.array(freqs)
	def read_beams(self):
		self.beams = np.zeros(self.nbands)
		f			= open(self.get_input('beams_list'),'r')
		for n,line in enumerate(f):
			self.beams[n] = float(line.strip())
	def read_mask(self):
		m			= hp.read_map(self.get_input('masks'),verbose=False)
		self.mask 	= m
		# We also need a binary mask where 0 stays 0, and >0 is 1
		self.bin_mask = np.ceil(self.mask)
	def read_ell_bins(self):
		bins_edges	= []
		f	= open(self.get_input('ell_bins_list'))
		for line in f:
			if line[0] == '#':
				continue
			else:
				bins_edges.append(int(line.strip()))
		self.bins_edges	= np.array(bins_edges)
	def read_freq_maps(self):
		# This is for reading the maps files and return it as a list of namaster fields
		f	= open(self.get_input('freq_maps_list'),'r')
		self.maps_s2_sp1 = []
		# beam transfer of the common channel
		b_common = hp.gauss_beam(np.radians(self.beams[self.beam_band]/60.0),lmax=(3*self.nside-1),pol=True)
		# when I read in the freq maps, I need to multiply the map
		for n,line in enumerate(f):
			if self.method == 'hilc':
				qu_list = hp.read_map(line.strip(), field=(1,2), verbose=False)
				self.maps_s2_sp1.append(nmt.NmtField(self.mask, [qu_list[0] , qu_list[1] ], purify_b=True,purify_e=True,masked_on_input=False ))
				print("Split 1 file %s done"%line.strip())
			elif self.method == 'nilc' or self.method == 'cnilc':
				# I need to transform the maps to alms in healpy
				tqu_list = hp.read_map(line.strip(), field=(0,1,2), verbose=False)
				alms = hp.map2alm(tqu_list,pol=True,iter=6,verbose=False)
				# I only keep the almsE and almsB
				# also I will upgrade resolution to the one of the highest channel 
				b_channel = hp.gauss_beam(np.radians(self.beams[n]/60.0),lmax=(3*self.nside-1),pol=True)
				almsE = hp.almxfl(alms[1],b_common[:,1]/b_channel[:,1])
				almsB = hp.almxfl(alms[2],b_common[:,2]/b_channel[:,2])
				self.maps_s2_sp1.append([almsE,almsB])
				print("Split 1 file %s done"%line.strip())
		self.maps_s2_sp1 = np.array(self.maps_s2_sp1)
	def load_ThermalDust_map(self):
		self.beta_dust_map = hp.read_map(self.get_input('beta_dust_map'),field=0,dtype=np.double)
		self.T_dust_map = hp.read_map(self.get_input('T_dust_map'),field=0,dtype=np.double)
	def calculate_covariance(self,alms_list,m_iter):
		# the covariance matrix will have shape (Nwindows,2,npix_per_window_max,Nfreq2). Since the npix per window is variable for each window, I will waste some memory, but it is easier to gather mpi
		covariance = {}
		for w in range(self.Nw):
			# we create an array with shape (2,Npix_per_window,Nfreq2)
			Nfreq_window = len(self.which_channels_which_bands[w])
			a = np.ones(Nfreq_window,dtype=np.double)
			npix_per_window = 12*self.nside_per_window[w]**2
			bin_mask_window = np.ceil(hp.ud_grade(self.bin_mask,self.nside_per_window[w]))
			Nfreqs2 = int(Nfreq_window*(Nfreq_window+1)/2) # Nfreqs2 is the number of cross frequencies, in lexicographic order
			cov_window  = np.zeros((2,npix_per_window,Nfreqs2),dtype=np.double)
			for field in range(2):
				tebFiltMaps = np.zeros((Nfreq_window,npix_per_window),dtype=np.double)
				for n,n_band in enumerate(self.which_channels_which_bands[w]):
					#print('Window ',w+1,' n=',n,' freq channel=',self.freqs[n_band])
					# this for only loops over the freq channels that contribute information to that needlet band
					# I have the alms as the input alms_list[freq][E or B]
					# Now I must filter for each needlet window
					alms_filt = hp.almxfl(alms_list[n_band,field,:],self.n_win[w])
					# In here, when I bring back to pixel map, I will do it at nside_per_window
					tebFiltMaps[n,:] = hp.alm2map(alms_filt,self.nside_per_window[w],pol=False,verbose=False)
					#print("filtering with window",w+1,"in band",n+1,"for field ",field)
				# now all the filtered freq maps are in teb. We smooth the cross maps for all frequency pairs according to the fwhm of each needlet window
				counter = 0
				for n in range(Nfreq_window):
					for nn in range(n,Nfreq_window):
						cov_window[field,:,counter] = hp.smoothing(tebFiltMaps[n,:]*tebFiltMaps[nn,:],fwhm=np.radians(self.fwhms[w]),pol=False,iter=6,verbose=False)
						#print("Window ",w+1," at freqs. ",n+1,nn+1," smoothing is done")
						counter += 1
			covariance[w] = cov_window
		# now I save the covariance dictionary to memory, it is easier this way
		covariance_file_out = self.get_output('covariance_pixel')+'.Iteration%04i'%(m_iter)
		pickle.dump(covariance, open(covariance_file_out,'wb'))
	def clean_nilc(self,alms_list,m_iter,save_weights=True):
		recons_cmb_final = np.zeros((2,self.npix),dtype=np.double)
		# we will run over these pixels
		# I will do one nilc window at a time
		for w in range(self.Nw):
			# each needlet window will be done with a sub-set of frequencies 
			Nfreq_window = len(self.which_channels_which_bands[w])
			#print('In window %i, I will work in the subset'%(w+1),self.freqs[self.which_channels_which_bands[w]])
			# now it is in thermo units, so a is an array of ones
			a = np.ones(Nfreq_window,dtype=np.double)
			#a = self.cmb(self.freqs[self.which_channels_which_bands[w]])
			#a = np.array(a / a[0],dtype=np.double) # a must be an array, even if it has a single element, this is for C 
			npix_per_window = 12*self.nside_per_window[w]**2
			bin_mask_window = np.ceil(hp.ud_grade(self.bin_mask,self.nside_per_window[w]))
			Nfreqs2 = int(Nfreq_window*(Nfreq_window+1)/2) # Nfreqs2 is the number of cross frequencies, in lexicographic order
			for field in range(2):
				#TEB2maps will be a numpy array with the shape [npix,Nfreqs2]
				TEB2maps = np.zeros((npix_per_window,Nfreqs2),dtype=np.double)
				tebFiltMaps = np.zeros((Nfreq_window,npix_per_window),dtype=np.double)
				for n,n_band in enumerate(self.which_channels_which_bands[w]):
					#print('Window ',w+1,' n=',n,' freq channel=',self.freqs[n_band])
					# this for only loops over the freq channels that contribute information to that needlet band
					# I have the alms as the input alms_list[freq][E or B]
					# Now I must filter for each needlet window
					alms_filt = hp.almxfl(alms_list[n_band,field,:],self.n_win[w])
					# In here, when I bring back to pixel map, I will do it at nside_per_window
					tebFiltMaps[n,:] = hp.alm2map(alms_filt,self.nside_per_window[w],pol=False,verbose=False)
					#print("filtering with window",w+1,"in band",n+1,"for field ",field)
				# now all the filtered freq maps are in teb. We smooth the cross maps for all frequency pairs according to the fwhm of each needlet window
				counter = 0
				for n in range(Nfreq_window):
					for nn in range(n,Nfreq_window):
						TEB2maps[:,counter] = hp.smoothing(tebFiltMaps[n,:]*tebFiltMaps[nn,:],fwhm=np.radians(self.fwhms[w]),pol=False,iter=6,verbose=False)
						#print("Window ",w+1," at freqs. ",n+1,nn+1," smoothing is done")
						counter += 1
				#print("First SHTs done for field ",field,"and window",w+1)
				indices = np.where(bin_mask_window == 1)
				pixels = indices[0].astype(np.uint) # this must be an array of C unsigned longs
				#pixels = np.arange(npix_per_window).astype('int32')
				w_ilc_tot = doNILC_SHTSmoothing_SingleField(TEB2maps, self.nside_per_window[w], a, Nfreq_window, pixels, len(pixels) , 0, field, self.Nthreads )
				if save_weights:
					# I save the weights because I will need it later to estimate the noise bias
					weights_file_out = self.get_output('NILC_weights')+'.Field%i-Window%i-NSIDE%i-Iteration%04i'%(field,w+1,self.nside_per_window[w],m_iter)
					np.savez(weights_file_out,w=w_ilc_tot)
				recons_cmb = np.zeros((npix_per_window),dtype=np.double)
				counter = 0
				for pi in range(npix_per_window):
					if bin_mask_window[pi] == 1.0:
						recons_cmb[pi] = np.matmul(w_ilc_tot[counter,:],tebFiltMaps[:,pi],dtype=np.double)
						counter += 1
				#print("reconstructed cmb for window %i at field %i ready"%(w+1,field))
				# We filter for the second time
				alms = hp.map2alm(recons_cmb,pol=False,iter=6)
				alms_filtered = hp.almxfl(alms,self.n_win[w])
				# finally, we sum over the windows
				#recons_cmb_final[field,:] += hp.alm2map(alms_filtered,self.nside,lmax=self.l_max,pol=False,verbose=False).astype(np.double)
				recons_cmb_final[field,:] += hp.alm2map(alms_filtered,self.nside,pol=False,verbose=False).astype(np.double)
				#print("Second SHTs done for window %i at field %i"%(w+1,field))
		return recons_cmb_final
	def mix_cmb_multi_nilc(self,alms_list,weights):
		recons_cmb_final = np.zeros((2,self.npix),dtype=np.double)
		# we will run over these pixels
		# I will do one nilc window at a time
		for w in range(self.Nw):
			# each needlet window will be done with a sub-set of frequencies 
			Nfreq_window = len(self.which_channels_which_bands[w])
			#print('In window %i, I will work in the subset'%(w+1),self.freqs[self.which_channels_which_bands[w]])
			# now it is in thermo units, so a is an array of ones
			a = np.ones(Nfreq_window,dtype=np.double)
			npix_per_window = 12*self.nside_per_window[w]**2
			bin_mask_window = np.ceil(hp.ud_grade(self.bin_mask,self.nside_per_window[w]))
			Nfreqs2 = int(Nfreq_window*(Nfreq_window+1)/2) # Nfreqs2 is the number of cross frequencies, in lexicographic order
			for field in range(2):
				tebFiltMaps = np.zeros((Nfreq_window,npix_per_window),dtype=np.double)
				for n,n_band in enumerate(self.which_channels_which_bands[w]):
					# this for only loops over the freq channels that contribute information to that needlet band
					# I have the alms as the input alms_list[freq][E or B]
					# Now I must filter for each needlet window
					alms_filt = hp.almxfl(alms_list[n_band,field,:],self.n_win[w])
					# In here, when I bring back to pixel map, I will do it at nside_per_window
					tebFiltMaps[n,:] = hp.alm2map(alms_filt,self.nside_per_window[w],pol=False,verbose=False)
				# now all the filtered freq maps are in teb. We smooth the cross maps for all frequency pairs according to the fwhm of each needlet window
				indices = np.where(bin_mask_window == 1)
				pixels = indices[0].astype(np.uint) # this must be an array of C unsigned longs
				recons_cmb = np.zeros((npix_per_window),dtype=np.double)
				counter = 0
				for pi in range(npix_per_window):
					if bin_mask_window[pi] == 1.0:
						recons_cmb[pi] = np.matmul(weights[w][field,counter,:],tebFiltMaps[:,pi],dtype=np.double)
						counter += 1
				#print("reconstructed cmb for window %i at field %i ready"%(w+1,field))
				# We filter for the second time
				alms = hp.map2alm(recons_cmb,pol=False,iter=6)
				alms_filtered = hp.almxfl(alms,self.n_win[w])
				# finally, we sum over the windows
				#recons_cmb_final[field,:] += hp.alm2map(alms_filtered,self.nside,lmax=self.l_max,pol=False,verbose=False).astype(np.double)
				recons_cmb_final[field,:] += hp.alm2map(alms_filtered,self.nside,pol=False,verbose=False).astype(np.double)
				#print("Second SHTs done for window %i at field %i"%(w+1,field))
		return recons_cmb_final
	def clean_cnilc(self,alms_list,m_iter,save_weights=True):
		recons_cmb_final = np.zeros((2,self.npix),dtype=np.double)
		# we will run over these pixels
		# I will do one nilc window at a time
		for w in range(self.Nw):
			# each needlet window will be done with a sub-set of frequencies 
			freq_arr_window = self.freqs[self.which_channels_which_bands[w]]
			Nfreq_window = len(freq_arr_window)
			#print('In window %i, I will work in the subset'%(w+1),self.freqs[self.which_channels_which_bands[w]])
			# now it is in thermo units, so a is an array of ones
			a = np.ones(Nfreq_window,dtype=np.double)
			#b = self.thermal_dust_thermo(self.freqs[self.which_channels_which_bands[w]],1.55,19.0)
			#a = self.cmb(self.freqs[self.which_channels_which_bands[w]])
			#a = np.array(a / a[0],dtype=np.double) # a must be an array, even if it has a single element, this is for C 
			npix_per_window = 12*self.nside_per_window[w]**2
			bin_mask_window = np.ceil(hp.ud_grade(self.bin_mask,self.nside_per_window[w]))
			Nfreqs2 = int(Nfreq_window*(Nfreq_window+1)/2) # Nfreqs2 is the number of cross frequencies, in lexicographic order
			for field in range(2):
				#TEB2maps will be a numpy array with the shape [npix,Nfreqs2]
				TEB2maps = np.zeros((npix_per_window,Nfreqs2),dtype=np.double)
				tebFiltMaps = np.zeros((Nfreq_window,npix_per_window),dtype=np.double)
				for n,n_band in enumerate(self.which_channels_which_bands[w]):
					#print('Window ',w+1,' n=',n,' freq channel=',self.freqs[n_band])
					# this for only loops over the freq channels that contribute information to that needlet band
					# I have the alms as the input alms_list[freq][E or B]
					# Now I must filter for each needlet window
					alms_filt = hp.almxfl(alms_list[n_band,field,:],self.n_win[w])
					# In here, when I bring back to pixel map, I will do it at nside_per_window
					tebFiltMaps[n,:] = hp.alm2map(alms_filt,self.nside_per_window[w],pol=False,verbose=False)
					#print("filtering with window",w+1,"in band",n+1,"for field ",field)
				# now all the filtered freq maps are in teb. We smooth the cross maps for all frequency pairs according to the fwhm of each needlet window
				counter = 0
				for n in range(Nfreq_window):
					for nn in range(n,Nfreq_window):
						TEB2maps[:,counter] = hp.smoothing(tebFiltMaps[n,:]*tebFiltMaps[nn,:],fwhm=np.radians(self.fwhms[w]),pol=False,iter=6,verbose=False)
						#print("Window ",w+1," at freqs. ",n+1,nn+1," smoothing is done")
						counter += 1
				#print("First SHTs done for field ",field,"and window",w+1)
				indices = np.where(bin_mask_window == 1)
				pixels = indices[0].astype(np.uint) # this must be an array of C unsigned longs
				#pixels = np.arange(npix_per_window).astype('int32')
				w_ilc_tot = doCNILC_ThermalDust_SHTSmoothing_SingleField(TEB2maps, self.nside_per_window[w], a, self.beta_dust_map, self.T_dust_map, freq_arr_window, Nfreq_window, pixels, len(pixels) , 0, field, self.Nthreads )
				if save_weights:
					# I save the weights because I will need it later to estimate the noise bias
					weights_file_out = self.get_output('NILC_weights')+'.Field%i-Window%i-NSIDE%i-Iteration%04i'%(field,w+1,self.nside_per_window[w],m_iter)
					np.savez(weights_file_out,w=w_ilc_tot)
				recons_cmb = np.zeros((npix_per_window),dtype=np.double)
				counter = 0
				for pi in range(npix_per_window):
					if bin_mask_window[pi] == 1.0:
						recons_cmb[pi] = np.matmul(w_ilc_tot[counter,:],tebFiltMaps[:,pi],dtype=np.double)
						counter += 1
				#print("reconstructed cmb for window %i at field %i ready"%(w+1,field))
				# We filter for the second time
				alms = hp.map2alm(recons_cmb,pol=False,iter=6)
				alms_filtered = hp.almxfl(alms,self.n_win[w])
				# finally, we sum over the windows
				#recons_cmb_final[field,:] += hp.alm2map(alms_filtered,self.nside,lmax=self.l_max,pol=False,verbose=False).astype(np.double)
				recons_cmb_final[field,:] += hp.alm2map(alms_filtered,self.nside,pol=False,verbose=False).astype(np.double)
				#print("Second SHTs done for window %i at field %i"%(w+1,field))
		return recons_cmb_final
	def read_nilc_windows(self):
		f_in = self.get_input('needlet_windows')
		self.n_win = np.loadtxt(f_in)
	def read_fwhms_dominion(self):
		f_in = self.get_input('fwhms_dominion')
		self.fwhms = np.loadtxt(f_in)
	def read_which_channels_which_bands(self):
		f_in = open(self.get_input('which_channels_which_bands'),'r')
		self.which_channels_which_bands = []
		for line in f_in:
			# each line is a needlet band
			arr = line.replace('\n','').split(',')
			x = list(map(int, arr)) # this transform the strings into ints
			self.which_channels_which_bands.append(x)
	def read_nside_per_window(self):
		f_in = self.get_input('nside_per_window')
		self.nside_per_window = np.loadtxt(f_in).astype(np.intc)
	def do_ILC_MonteCarlo(self):
		# First I need to read the two sets of splits
		f	= open(self.get_input('freq_maps_list_MC'),'r')
		f2	= open(self.get_input('freq_maps_list_MC'),'r')
		self.weights_MC = np.zeros((self.Nmc,2,self.nbands,3*self.nside))
		self.cls_cmb_MC = np.zeros((self.Nmc,2,3*self.nside))
		for m in range(self.Nmc):
			alms_s2_sp1 = [] 
			alms_s2_sp2 = []
			for n in range(self.nbands):
				file1 = f.readline().strip()
				file2 = f2.readline().strip()
				qu_list_sp1 = hp.read_map(file1, field=(1,2), verbose=False)
				alms_s2_sp1.append(nmt.NmtField(self.mask, [qu_list_sp1[0] , qu_list_sp1[1] ], purify_b=True,purify_e=True,masked_on_input=False))
				if self.weights_option=='cross':
					qu_list_sp2 = hp.read_map(file2, field=(1,2), verbose=False)
					alms_s2_sp2.append(nmt.NmtField(self.mask, [qu_list_sp2[0] , qu_list_sp2[1] ], purify_b=True,purify_e=True,masked_on_input=False))
				print("Iter=%i freq. %.1f GHz done with files %s and %s"%(m,self.freqs[n],file1,file2))
			if self.weights_option=='cross':
				resultP 	= self.clean_ilc(alms_s2_sp1,alms_s2_sp2,pol=True)
			elif self.weights_option=='auto':
				resultP 	= self.clean_ilc(alms_s2_sp1,alms_s2_sp1,pol=True)
			self.weights_MC[m] = resultP['w_ilc']
			self.cls_cmb_MC[m] = resultP['cl_c']
			print("Iter=%i done"%m)
	def write_ILC_MonteCarlo(self):
		fname = self.get_output('ilc_weights_MC')
		np.savez(fname,ilc_weights_MC=self.weights_MC,Nmc=self.Nmc)
	def write_ILC_spectra(self):
		fname = self.get_output('cl_cmb_MC')
		np.savez(fname,cls_cmb_MC=self.cls_cmb_MC,Nmc=self.Nmc)
	def thermo2RJ(self,nu):
		h_p =  6.6260755e-34
		k_b = 1.380658e-23
		T_cmb = 2.72548
		x_cmb =  h_p * nu *1e9 / k_b / T_cmb
		sed = x_cmb**2 * np.exp(x_cmb) / (np.exp(x_cmb) - 1.0)**2
		return sed
	def thermal_dust_thermo(self,nu,beta,T):
		h_p =  6.6260755e-34
		k_b = 1.380658e-23
		xd = h_p*nu*1e9/(k_b*T)
		return nu**(beta+1)/(np.exp(xd)-1) / self.thermo2RJ(nu)
	def clean_ilc(self,alms_list,alms_list2,pol):
		nside		= self.nside
		verbose		= self.verbose
		freqs		= self.freqs
		ell_bins	= self.bins_edges
		
		def pspec_compute(alm1,alm2) :
			cls=nmt.compute_coupled_cell(alm1,alm2)
			if len(alm1.get_maps())==2 :
				return np.array([cls[0],cls[3],cls[1]])
			else :
				return cls
		n_nu=len(freqs)
		c_nu=self.cmb(freqs) #Frequency dependence
		if len(alms_list)!=len(c_nu) or len(alms_list2)!=len(c_nu) :
			raise ValueError("#maps != #freqs")
		n_ell=3*nside
		if ell_bins[-1]!=n_ell :
			raise ValueError("Bins don't include all ells")
		bands=np.transpose(np.array([ell_bins[:-1],ell_bins[1:]]))
		n_bands=len(bands)
		if pol :
			n_pols=2
		else :
			n_pols=1
		#Compute power spectra
		if verbose :
			print("Power spectra")
		c_ells=np.zeros([n_pols,n_nu,n_nu,n_ell])
		for inu1 in np.arange(n_nu) :
			for inu2 in np.arange(inu1,n_nu) :
				cls=pspec_compute(alms_list[inu1],alms_list2[inu2])
				for ip in np.arange(n_pols) :
					c_ells[ip,inu1,inu2]=cls[ip]
					if inu2!=inu1 :
						c_ells[ip,inu2,inu1]=cls[ip]
		#Compute ILC weights
		if verbose :
			print("Weights")
		w_ilc=np.zeros([n_pols,n_nu,n_ell])
		for ib in np.arange(n_bands) :
			l_list=np.arange(bands[ib,0],bands[ib,1])
			lweight=(2*l_list+1.)/np.sum(2*l_list+1.)
			for ip in np.arange(n_pols) :
				covar=np.sum(c_ells[ip][:,:,l_list]*lweight,axis=2)
				covm1_c=np.linalg.solve(covar,c_nu)
				norm=1./np.dot(c_nu,covm1_c)
				w_ilc[ip][:,l_list]=(covm1_c*norm)[:,None]

		#Project onto weights (and do the same for the noise bias) 
		if verbose :
			print("Projection")
		cl_c=np.sum(w_ilc*np.sum(c_ells*w_ilc[:,None,:,:],axis=2),axis=1)

		return {'cl_c':cl_c,'w_ilc':w_ilc,'cl_nu':c_ells,'bands':bands}

	def save_cl_to_file(self,cl,fname):
		print('Saving to file '+fname)
		hp.write_cl(fname,cl,overwrite=True)
	
	def save_ilcweights_to_file(self,ilc_weights_T,ilc_weights_P,fname):
		print('Saving to file'+fname)
		np.savez(fname,ilc_weights_T=ilc_weights_T,ilc_weights_P=ilc_weights_P)
		
	def run(self):
		# setup the mpi communicator
		comm = MPI.COMM_WORLD
		size_pool = comm.Get_size()
		rank = comm.Get_rank()
		
		self.init_params()
		self.read_freqs()
		self.read_beams()
		self.read_mask()
		self.read_ell_bins()
		#if rank == 0:
		#	self.read_freq_maps()
		if self.method=='hilc':
			if self.weights_option=='cross':
				resultP 	= self.clean_ilc(self.maps_s2_sp1,self.maps_s2_sp2,pol=True)
			elif self.weights_option=='auto':
				resultP 	= self.clean_ilc(self.maps_s2_sp1,self.maps_s2_sp1,pol=True)
			np.savez(self.get_output('bands'),bands = resultP['bands'])
			# join the spectra
			cls_cmb			= [np.zeros_like(resultP['cl_c'][0]),resultP['cl_c'][0],resultP['cl_c'][1]]
			# save cell cmb to healpy fits file
			self.save_cl_to_file(cls_cmb,self.get_output('cl_cmb'))
			# save ilc weights
			self.save_ilcweights_to_file(resultP['w_ilc'],resultP['w_ilc'],self.get_output('ilc_weights'))
			# do the MC iterations
			self.do_ILC_MonteCarlo()
			self.write_ILC_MonteCarlo()
			self.write_ILC_spectra()
		elif self.method=='nilc' or self.method=='cnilc':
			if True:
				self.read_which_channels_which_bands()
				self.read_nilc_windows()
				self.read_fwhms_dominion()
				self.read_nside_per_window()
				if self.method=='cnilc': self.load_ThermalDust_map()
				# do MC nilc, we will do it with mpi
				if True:
					if rank==0:
						mc_indices = np.arange(0,self.Nmc,dtype='int32')
						mc_indices_split = np.array_split(mc_indices, size_pool)
						# because I don't know how many indices each rank will get, I need to send the shapes first
						shapes = []
						for rank_id in range(size_pool):
							shapes.append(len(mc_indices_split[rank_id]))
						split_sizes_output = np.array(shapes)*2*self.npix
						displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
					else:
						shapes = None
						split_sizes_output = None
						displacements_output = None
					shapes = comm.bcast(shapes, root=0) # broadcast the shapes to each rank
					split_sizes_output = comm.bcast(split_sizes_output, root = 0)
					displacements_output = comm.bcast(displacements_output, root = 0)
					if rank==0:
						# I go over each rank from 1 to (poolsize-1)
						for rank_id in range(1,size_pool):
							# this mask is the indices of the filaments that correspond to rank=rank_id
							mask = mc_indices_split[rank_id]
							comm.Send([mc_indices_split[rank_id], MPI.INT], dest=rank_id, tag=1) # indices
						mc_indices_split_rank = mc_indices_split[rank]
					else:
						mc_indices_split_rank = np.empty(shapes[rank], dtype='int32')
						comm.Recv([mc_indices_split_rank, MPI.INT], source=0, tag=1)
					# put a barrier so that every rank is done on receiving their indices to run
					comm.Barrier()
					# read the file with all the filenames of the Nmc maps
					f	= open(self.get_input('freq_maps_list_MC'),'r')
					filenames = f.readlines()
					# now I create an array with size shape[rank] to hold the E and B map
					cmb_EB_MC = np.zeros((shapes[rank],2,self.npix),dtype=np.double)
					# All transfer functions for the freq. channels
					beams_transfer = []
					for n in range(self.nbands):
						beams_transfer.append(hp.gauss_beam(np.radians(self.beams[n]/60.0),lmax=(3*self.nside-1),pol=True))
					# this array is to keep the alms of the input maps which I need for the mixing later, so I don't have to recalculate
					alms_s2_array = []
					for m,m_rank in enumerate(mc_indices_split_rank):
						alms_s2_sp1 = []
						for n in range(self.nbands):
							file1 = filenames[m_rank*6+n].replace('\n','')
							tqu_list_sp1 = hp.read_map(file1, field=(0,1,2), verbose=False)
							# here, the alms need to be up to the maximum nside in self.nside_per_window, this will save some memory
							alms = hp.map2alm(tqu_list_sp1,pol=True,iter=6,verbose=False,lmax=(3*np.max(self.nside_per_window)-1))
							almsE = hp.almxfl(alms[1],beams_transfer[self.beam_band][:,1]/beams_transfer[n][:,1])
							almsB = hp.almxfl(alms[2],beams_transfer[self.beam_band][:,2]/beams_transfer[n][:,2])
							alms_s2_sp1.append([almsE,almsB])
							#print("MC iteration=%i on rank %i, freq. %.1f GHz done with file %s"%(m_rank,rank,self.freqs[n],file1))
						alms_s2_sp1 = np.array(alms_s2_sp1) ; 
						alms_s2_array.append(alms_s2_sp1)
						if self.method=='nilc':
							if self.weights_mode == 'single':
								cmb_EB_MC[m,:,:] = self.clean_nilc(alms_s2_sp1,m_rank)
							elif self.weights_mode == 'multi':
								self.calculate_covariance(alms_s2_sp1,m_rank)
						elif self.method=='cnilc':
							cmb_EB_MC[m,:,:] = self.clean_cnilc(alms_s2_sp1,m_rank)
						print("MC iteration=%i is done on rank %i"%(m_rank,rank))
					comm.Barrier() # I have to wait for all ranks to be done calculating the covariances
					if self.weights_mode == 'multi':
						# if weights_mode is multi, then after I run all realizations I load the individual iteration covariances and average over all realizations
						covariance_final = {}
						for w in range(self.Nw):
							# I create an empty array for window w to hold the mean covariance
							freq_arr_window = self.freqs[self.which_channels_which_bands[w]]
							Nfreq_window = len(freq_arr_window)
							Nfreqs2 = int(Nfreq_window*(Nfreq_window+1)/2)
							npix_per_window = 12*self.nside_per_window[w]**2
							covariance_mean = np.zeros((2,npix_per_window,Nfreqs2),dtype=np.double)
							for m in range(self.Nmc):
								# load the covariance file for iteration m
								covariance_file_in = self.get_output('covariance_pixel')+'.Iteration%04i'%(m)
								covariance_mean += pickle.load(open(covariance_file_in,'rb'))[w]
							# after we cycle over Nmc iterations, we take the average over axis = 0
							covariance_final[w] = covariance_mean / self.Nmc
						covariance_mean = None # I don't need it anymore
						# Now I calculate the unique weights, once per rank
						weights = {}
						for w in range(self.Nw):
							# each needlet window will be done with a sub-set of frequencies 
							Nfreq_window = len(self.which_channels_which_bands[w])
							#print('In window %i, I will work in the subset'%(w+1),self.freqs[self.which_channels_which_bands[w]])
							# now it is in thermo units, so a is an array of ones
							a = np.ones(Nfreq_window,dtype=np.double)
							npix_per_window = 12*self.nside_per_window[w]**2
							bin_mask_window = np.ceil(hp.ud_grade(self.bin_mask,self.nside_per_window[w]))
							Nfreqs2 = int(Nfreq_window*(Nfreq_window+1)/2) # Nfreqs2 is the number of cross frequencies, in lexicographic order
							# the shape of weights from doNILC_SHTSmoothing_SingleField is (npix_per_window,Nfreq_window)
							indices = np.where(bin_mask_window == 1)
							pixels = indices[0].astype(np.uint)
							w_ilc_tot_fields = np.zeros((2,len(pixels),Nfreq_window),dtype=np.double)
							for field in range(2):
								w_ilc_tot_fields[field] = doNILC_SHTSmoothing_SingleField(covariance_final[w][field,:,:], self.nside_per_window[w], a, Nfreq_window, pixels, len(pixels) , 0, field, self.Nthreads)
							weights[w] = w_ilc_tot_fields
							w_ilc_tot_fields = None
						del covariance_final
						gc.collect() # this is to release the memory of covariance_final
						# now weights contains the unique weights to mix all realizations
						if rank == 0:
							# save it to a file in rank 0
							weights_file_out = self.get_output('ilc_weights_multi')
							pickle.dump(weights, open(weights_file_out,'wb'))
						for m,m_rank in enumerate(mc_indices_split_rank):
							alms_s2_sp1 = alms_s2_array[m]
							if self.method=='nilc':
								cmb_EB_MC[m,:,:] = self.mix_cmb_multi_nilc(alms_s2_sp1,weights)
					# after every rank is done, I Gather onto a single array with shape Nmc
					comm.Barrier()
					if rank == 0:
						cmb_EB_MC_final = np.zeros((self.Nmc,2,self.npix),dtype=np.double)
					else:
						cmb_EB_MC_final = None
					comm.Gatherv(cmb_EB_MC,[cmb_EB_MC_final,split_sizes_output,displacements_output,MPI.DOUBLE], root=0)
					# save to a numpy npz file
					f1 = self.get_output('reconstruced_cmb_MC')
					np.savez(f1,cmb_EB_MC=cmb_EB_MC_final)
if __name__ == '__main__':
	cls = PipelineStage.main()
