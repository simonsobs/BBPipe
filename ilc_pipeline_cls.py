import numpy as np
import healpy as hp
import pymaster as nmt
from bbpipe import PipelineStage
import pickle
from types_mine import FitsFile,TextFile,DummyFile,NpzFile,PklFile

class BB_Cl_Process(PipelineStage):
	name 			= 'BB_Cl_Process'
	inputs			= [('noise_maps_list_MC',TextFile),('cl_cmb',FitsFile),('cl_cmb_MC',NpzFile),('ilc_weights',NpzFile),('ell_bins_list',TextFile),('freqs_list',TextFile),('masks',TextFile),('reconstruced_cmb_MC',NpzFile),('NILC_weights',NpzFile),('ilc_weights_multi',PklFile),('needlet_windows',TextFile),('fwhms_dominion',TextFile),('nside_per_window',TextFile),('which_channels_which_bands',TextFile),('beams_list',TextFile)]
	outputs			= [('bandpowers_MC',NpzFile),(('noise_bias',NpzFile)),('NmtWorkspace',FitsFile)]
	def init_params(self):
		self.nside 		= self.config['nside']
		self.npix 		= hp.nside2npix(self.nside)
		self.pol		= self.config['pol']
		self.verbose	= self.config['verbose']
		self.Nmc		= self.config['Nsimul']
		self.Nnoise = self.config['Nnoise']
		self.method = self.config['method']
		self.beam_band = int(self.config['beam_band'])
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
		self.Nfreqs	= len(self.freqs)
	def read_mask(self):
		m			= hp.read_map(self.get_input('masks'),verbose=False)
		self.mask 	= m
		# We also need a binary mask where 0 stays 0, and >0 is 1
		self.bin_mask = np.ceil(self.mask)
	def read_beams(self):
		self.beams = np.zeros(self.nbands)
		f			= open(self.get_input('beams_list'),'r')
		for n,line in enumerate(f):
			self.beams[n] = float(line.strip())
	def read_ell_bins(self):
		bins_edges = np.loadtxt(self.get_input('ell_bins_list'))
		bins_edges_min = [] ; bins_edges_max = []
		for n in range(len(bins_edges)-1):
			bins_edges_min.append(int(bins_edges[n]))
			bins_edges_max.append(int(bins_edges[n+1]))
		self.bins = nmt.NmtBin.from_edges(bins_edges_min, bins_edges_max, is_Dell=True)
		self.Nbins = self.bins.get_n_bands()
		print('The bins will be')
		for n in range(self.Nbins):
			print('bin %i ell_min %i ell_max %i'%(n,self.bins.get_ell_min(n),self.bins.get_ell_max(n)))
	def read_nilc_windows(self):
		f_in = self.get_input('needlet_windows')
		self.n_win = np.loadtxt(f_in)
	def read_fwhms_dominion(self):
		f_in = self.get_input('fwhms_dominion')
		self.fwhms = np.loadtxt(f_in)
	def read_nside_per_window(self):
		f_in = self.get_input('nside_per_window')
		self.nside_per_window = np.loadtxt(f_in).astype(np.intc)
		#print(self.nside_per_window)
	def read_which_channels_which_bands(self):
		f_in = open(self.get_input('which_channels_which_bands'),'r')
		self.which_channels_which_bands = []
		for line in f_in:
			# each line is a needlet band
			arr = line.replace('\n','').split(',')
			x = list(map(int, arr)) # this transform the strings into ints
			self.which_channels_which_bands.append(x)
	def read_cmb_cls(self):
		cls_cmb			= hp.read_cl(self.get_input('cl_cmb'))
		# Fill in 
		self.nels = cls_cmb.shape[1]
		self.cmb_cls = np.zeros([4,self.nels])
		# EE is in index 0, BB is in index 3
		# if we are in auto weights, this is where we subtract the noise bias
		if self.weights_option=='auto':
			self.cmb_cls[0,:]	= cls_cmb[1] - self.noise_bias[0]
			self.cmb_cls[3,:]	= cls_cmb[2] - self.noise_bias[1]
		elif self.weights_option=='cross':
			self.cmb_cls[0,:]	= cls_cmb[1] 
			self.cmb_cls[3,:]	= cls_cmb[2]
	def run_decoupling_MonteCarlo(self):
		self.cls_cmb_MC = np.load(self.get_input('cl_cmb_MC'))['cls_cmb_MC']
		self.bandpowers_MC = np.zeros((self.Nmc,2,self.Nbins))
		# define the cls in the spin-2 shape
		for m in range(self.Nmc):
			cls_for_decoupling = np.zeros([4,self.nels])
			if self.weights_option=='auto':
				cls_for_decoupling[0,:] = self.cls_cmb_MC[m,0,:] - self.noise_bias[0]
				cls_for_decoupling[3,:] = self.cls_cmb_MC[m,1,:] - self.noise_bias[1]
			elif self.weights_option=='cross':
				cls_for_decoupling[0,:] = self.cls_cmb_MC[m,0,:]
				cls_for_decoupling[3,:] = self.cls_cmb_MC[m,1,:]
			bandpowers = self.compute_bandpowers(cls_for_decoupling)
			self.bandpowers_MC[m,0,:] = bandpowers[0,:]
			self.bandpowers_MC[m,1,:] = bandpowers[3,:]
			print("Decoupling for iteration %i is ready"%m)
	def estimate_noise_bias(self):
		# first define pspec
		f	= open(self.get_input('noise_maps_list_MC'),'r')
		if self.method == 'hilc':
			def pspec_compute(alm1,alm2) :
				cls=nmt.compute_coupled_cell(alm1,alm2)
				if len(alm1.get_maps())==2 :
					return np.array([cls[0],cls[3],cls[1]])
				else :
					return cls
			n_ells		= np.zeros([self.Nnoise,2,self.nbands,self.nbands,3*self.nside])
			nl_c_array_P	= np.zeros((self.Nnoise,2,3*self.nside))
			# I need to load the ilc weights
			ilc_w_P = np.load(self.get_input('ilc_weights'))['ilc_weights_P']
			print('Estimating noise bias')
			for m in range(self.Nnoise):
				noise_alms_s2 = []
				for n in range(self.nbands):
					line = f.readline()
					qu_list = hp.read_map(line.strip(), field=(1,2), verbose=False)
					noise_alms_s2.append(nmt.NmtField(self.mask, [qu_list[0] , qu_list[1] ], purify_b=True,purify_e=True,masked_on_input=False ))
				# now I have the alms for realization m
				for inu1 in range(self.nbands):
					for inu2 in range(inu1,self.nbands):
						noi_cls = pspec_compute(noise_alms_s2[inu1],noise_alms_s2[inu2])
						for ip in np.arange(2) :
							n_ells[m,ip,inu1,inu2,:]=noi_cls[ip]
							# if we are not in the diagonal of n_ells, we copy the symmetric element
							if inu1!=inu2: n_ells[m,ip,inu2,inu1,:]=noi_cls[ip]
				nl_c = np.sum(ilc_w_P*np.sum(n_ells[m,:,:,:,:]*ilc_w_P[:,None,:,:],axis=2),axis=1)
				nl_c_array_P[m,:,:] = nl_c
				print('Done with noise sim %i out of %i'%(m+1,self.Nnoise))
			nl_c_mean = np.mean(nl_c_array_P,axis=0)
			self.noise_bias = nl_c_mean
		elif self.method == 'nilc' or self.method == 'cnilc':
			print("Starting noise bias calculation")
			# in this array I will keep the decoupled power spectra
			# this has shape 4 because the spin-2 spectra in namaster are 4  
			nl_c_array_P	= np.zeros((self.Nnoise,4,3*self.nside),dtype=np.double)
			# beams
			beams_transfer = []
			for n in range(self.nbands):
				beams_transfer.append(hp.gauss_beam(np.radians(self.beams[n]/60.0),lmax=(3*self.nside-1),pol=True))
			# if we use the multi averaged covariance matrix, then the weights are unique
			if self.weights_mode == 'multi':
				# you only need to load the weights a single time
				w_file = self.get_input('ilc_weights_multi')
				weights_dict = pickle.load(open(w_file,'rb'))
			for m in range(self.Nnoise):
				alms_list = []
				for n in range(self.nbands):
					line = f.readline()
					tqu_map = hp.read_map(line.strip(), field=(0,1,2), verbose=False)
					# this is the tqu map of noise, we need to transform to E B maps
					alms = hp.map2alm(tqu_map,pol=True,iter=6,verbose=False)
					# I smooth the alms to a common resolution of the channel self.beam_band
					almsE = hp.almxfl(alms[1],beams_transfer[self.beam_band][:,1]/beams_transfer[n][:,1])
					almsB = hp.almxfl(alms[2],beams_transfer[self.beam_band][:,2]/beams_transfer[n][:,2])
					alms_list.append([almsE,almsB])
				# now I have the E and B alms for each band for iteration m
				alms_list = np.array(alms_list)
				recons_noise_final  = np.zeros((2,self.npix),dtype=np.double)
				for w in range(self.Nw):
					Nfreq_window = len(self.which_channels_which_bands[w])
					npix_per_window = 12*self.nside_per_window[w]**2
					bin_mask_window = np.round(hp.ud_grade(self.bin_mask,self.nside_per_window[w]))
					Nfreqs2 = int(Nfreq_window*(Nfreq_window+1)/2) # Nfreqs2 is the number of cross frequencies, in lexicographic order
					for field in range(2):
						tebFiltMaps = np.zeros((Nfreq_window,npix_per_window),dtype=np.double)
						for n,n_band in enumerate(self.which_channels_which_bands[w]):
							# I have the alms as the input alms_list[freq][E or B]
							# Now I must filter for each needlet window
							alms_filt = hp.almxfl(alms_list[n_band,field,:],self.n_win[w])
							tebFiltMaps[n,:] = hp.alm2map(alms_filt,self.nside_per_window[w],pol=False,verbose=False)
							#print("filtering with window",w+1,"in band",n+1,"for field ",field)
						#print("First SHTs done for field ",field,"and window",w+1)
						indices = np.where(bin_mask_window == 1)
						pixels = indices[0].astype('int32')
						if self.weights_mode == 'single':
							# I load the weights calculated in the first stage
							w_file = self.get_input('NILC_weights')+'.Field%i-Window%i-NSIDE%i-Iteration%04i.npz'%(field,w+1,self.nside_per_window[w],m)
							w_ilc_tot = np.load(w_file)['w']
						elif self.weights_mode == 'multi':
							w_ilc_tot = weights_dict[w][field,:,:]
						recons_cmb = np.zeros((npix_per_window),dtype=np.double)
						counter = 0
						for pi in range(npix_per_window):
							if bin_mask_window[pi] == 1.0:
								recons_cmb[pi] = np.matmul(w_ilc_tot[counter,:],tebFiltMaps[:,pi],dtype=np.double)
								counter += 1
						#print("reconstructed noise for window %i at field %i ready"%(w+1,field))
						# We filter for the second time
						alms = hp.map2alm(recons_cmb,pol=False,iter=6)
						alms_filtered = hp.almxfl(alms,self.n_win[w])
						# finally, we sum over the windows
						recons_noise_final[field,:] += hp.alm2map(alms_filtered,self.nside,pol=False,verbose=False).astype(np.double)
						#print("Second SHTs done for window %i at field %i"%(w+1,field))
				# now I have the reconstructed noise map, and I need to estimate the power spectra
				# I will transform the E/B maps into Q/U so I can define a spin-2 NmtField with purify_b = True
				# this has the common beam from band self.beam_band, so we have to include it in the NmtField
				# this is in CMB units
				almsE = hp.map2alm(recons_noise_final[0],iter=6,pol=False,verbose=False)
				almsB = hp.map2alm(recons_noise_final[1],iter=6,pol=False,verbose=False)
				# The size of a alms is size = mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1. lmax = mmax in general
				lmax = 3*self.nside - 1 ; mmax = lmax
				size = int(mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1)
				almsT = np.zeros((size),dtype=np.complex128)
				TQU_map = hp.alm2map((almsT,almsE,almsB),self.nside,pol=True,verbose=False)
				f_s2 = nmt.NmtField(self.mask, [TQU_map[1],TQU_map[2]], spin=2, masked_on_input=False, n_iter=6,beam=beams_transfer[self.beam_band][:,2],purify_b=True,purify_e=True)
				nl_c_array_P[m] = nmt.compute_coupled_cell(f_s2,f_s2)
				print('Done with noise sim %i out of %i'%(m+1,self.Nnoise))
			# now I have ran all Nnoise iterations, I take the mean across the Monte Carlo iterations to get an estimated noise bias
			self.noise_bias = np.mean(nl_c_array_P,axis=0)
	def read_cmb_map_and_calculateCell(self):
		# I will transform the E/B maps into Q/U so I can define a spin-2 NmtField with purify_b = True
		# this has the common beam from band self.beam_band, so we have to include it in the NmtField
		# this is in CMB units
		beam_common = hp.gauss_beam(np.radians(self.beams[self.beam_band]/60.),lmax=(3*self.nside-1),pol=True)
		cmb_map_sp1 = hp.read_map(self.get_input('reconstructed_cmb'),field=(0,1))
		almsE = hp.map2alm(cmb_map_sp1[0],iter=6,pol=False,verbose=False)
		almsB = hp.map2alm(cmb_map_sp1[1],iter=6,pol=False,verbose=False)
		# The size of a alms is size = mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1. lmax = mmax in general
		lmax = 3*self.nside - 1 ; mmax = lmax
		size = int(mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1)
		almsT = np.zeros((size),dtype=np.complex128)
		TQU_map = hp.alm2map((almsT,almsE,almsB),self.nside,pol=True,verbose=False)
		f_s2 = nmt.NmtField(self.mask, [TQU_map[1],TQU_map[2]], spin=2, masked_on_input=False, n_iter=6,beam=beam_common[:,2],purify_b=True,purify_e=True)
		self.workspace	= nmt.NmtWorkspace()
		self.workspace.compute_coupling_matrix(f_s2,f_s2,bins=self.bins,is_teb=False,n_iter=6)
		# save the workspace object to a fits file
		file_workspace = self.get_output('NmtWorkspace')
		self.workspace.write_to(file_workspace)
		cls_for_decoupling = nmt.compute_coupled_cell(f_s2,f_s2)
		decoupled_cl = self.workspace.decouple_cell(cls_for_decoupling,cl_noise=self.noise_bias)
		# I also bin and write the noise bias
		noise_bias = np.zeros((2,self.Nbins))
		decoupled_noise = self.workspace.decouple_cell(self.noise_bias)
		noise_bias[0,:] = decoupled_noise[0]
		noise_bias[1,:] = decoupled_noise[3]
		fname = self.get_output('noise_bias')
		np.savez(fname, noise_bias = noise_bias)
		# return EE (index 0) and BB (index 3)
		return decoupled_cl[0],decoupled_cl[3]
	def write_bandpowers_MC(self):
		fname = self.get_output('bandpowers_MC')
		np.savez(fname,bandpowers_MC=self.bandpowers_MC,Nmc=self.Nmc)
	def create_nmt_Workspace(self):
		# this method is only used for HILC
		# This just sets a nmt workspace
		self.workspace	= nmt.NmtWorkspace()
		self.workspace.compute_coupling_matrix(self.SingleMap_s2,self.SingleMap_s2,bins=self.bins,is_teb=False)
	def compute_bandpowers(self,cmb_cls):
		# cmb_cls has TT,EE and BB spectra
		bandpowers	= self.workspace.decouple_cell(cmb_cls)
		# We also do the noise bias at the same time
		noise_bias_to_decouple = np.zeros((4,self.nels))
		noise_bias_to_decouple[0] = self.noise_bias[0]
		noise_bias_to_decouple[3] = self.noise_bias[1]
		noise_bias_binned = self.workspace.decouple_cell(noise_bias_to_decouple)
		# save to a npz file
		fname = self.get_output('noise_bias')
		np.savez(fname,noise_bias = noise_bias_binned)
		return bandpowers
	def save_bandpowers_to_file(self,cl,fname):
		print('Saving to file ...'+fname)
		f		= open(fname,'w')
		Nbin	= self.bins.get_n_bands()
		for i in range(Nbin):
			ells_in_bin	= self.bins.get_ell_list(i)
			ell_low		= ells_in_bin[0]
			ell_high	= ells_in_bin[-1]
			f.write('%i\t%i\t%E\t%E\n'%(ell_low,ell_high,cl[0,i],cl[3,i]))
		f.close()
	def read_cmb_map_and_calculateCell_MC(self):
		# This has shape 2 because one for EE and one for BB
		self.bandpowers_MC = np.zeros((self.Nmc,4,self.Nbins))
		# load the cmb maps from the npz file
		cmb_EB_MC_sp1 = np.load(self.get_input('reconstruced_cmb_MC'))['cmb_EB_MC']
		beam_common = hp.gauss_beam(np.radians(self.beams[self.beam_band]/60.),lmax=(3*self.nside-1),pol=True)
		for m in range(self.Nmc):
			almsE = hp.map2alm(cmb_EB_MC_sp1[m,0],iter=6,pol=False,verbose=False)
			almsB = hp.map2alm(cmb_EB_MC_sp1[m,1],iter=6,pol=False,verbose=False)
			# The size of a alms is size = mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1. lmax = mmax in general
			lmax = 3*self.nside - 1 ; mmax = lmax
			size = int(mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1)
			almsT = np.zeros((size),dtype=np.complex128)
			TQU_map = hp.alm2map((almsT,almsE,almsB),self.nside,pol=True,verbose=False)
			f_s2 = nmt.NmtField(self.mask, [TQU_map[1],TQU_map[2]], spin=2, masked_on_input=False, n_iter=6,beam=beam_common[:,2],purify_b=True,purify_e=True)
			cls_for_decoupling = nmt.compute_coupled_cell(f_s2,f_s2)
			if m == 0:
				self.workspace	= nmt.NmtWorkspace()
				self.workspace.compute_coupling_matrix(f_s2,f_s2,bins=self.bins,is_teb=False,n_iter=6)
				# save the workspace object to a fits file
				file_workspace = self.get_output('NmtWorkspace')
				self.workspace.write_to(file_workspace)
				# I also write the decoupled noise bias to a file
				# I also bin and write the noise bias
				noise_bias = np.zeros((2,self.Nbins))
				decoupled_noise = self.workspace.decouple_cell(self.noise_bias)
				noise_bias[0,:] = decoupled_noise[0]
				noise_bias[1,:] = decoupled_noise[3]
				fname = self.get_output('noise_bias')
				np.savez(fname, noise_bias = noise_bias)
			decoupled_cl = self.workspace.decouple_cell(cls_for_decoupling,cl_noise=self.noise_bias)
			self.bandpowers_MC[m,0,:] = decoupled_cl[0]
			self.bandpowers_MC[m,1,:] = decoupled_cl[3]
			print('Iteration %i done'%m)
	def run(self):
		self.init_params()
		self.read_freqs()
		self.read_mask()
		self.read_ell_bins()
		self.read_beams()
		if self.method=='hilc':
			# coupling matrix and binning
			if self.weights_option=='auto':
				self.estimate_noise_bias()
			self.read_cmb_cls()
			self.create_nmt_Workspace()
			bandpowers	= self.compute_bandpowers(self.cmb_cls)
			self.save_bandpowers_to_file(bandpowers,self.get_output('bandpowers'))
			# Now run the decoupling for the Nmc spectra
			self.run_decoupling_MonteCarlo()
			self.write_bandpowers_MC()
		elif self.method=='nilc' or self.method=='cnilc':
			self.read_nilc_windows()
			self.read_fwhms_dominion()
			self.read_nside_per_window()
			self.read_which_channels_which_bands()
			# estimate the noise bias
			self.estimate_noise_bias()
			# calculate power spectra of cmb map
			#cl_ee, cl_bb = self.read_cmb_map_and_calculateCell()
			# save to file
			#fname = self.get_output('bandpowers')
			#print('Saving to file ...'+fname)
			#		= open(fname,'w')
			#Nbin	= self.bins.get_n_bands()
			#for i in range(Nbin):
			#	ells_in_bin	= self.bins.get_ell_list(i)
			#	ell_low		= ells_in_bin[0]
			#	ell_high	= ells_in_bin[-1]
			#	f.write('%i\t%i\t%E\t%E\n'%(ell_low,ell_high,cl_ee[i],cl_bb[i]))
			#f.close()
			# do the decoupling for MC
			self.read_cmb_map_and_calculateCell_MC()
			self.write_bandpowers_MC()
if __name__ == '__main__':
	cls = PipelineStage.main()
