import numpy as np
import healpy as hp
import pymaster as nmt
from bbpipe import PipelineStage
from types_mine import FitsFile,TextFile,DummyFile,NpzFile

class BB_Cl_Process(PipelineStage):
	name 			= 'BB_Cl_Process'
	inputs			= [('freq_maps_list_sp1',TextFile),('cl_cmb',FitsFile),('cl_cmb_MC',NpzFile),('ilc_weights',NpzFile),('ell_bins_list',TextFile),('freqs_list',TextFile),('masks',TextFile)]
#	outputs			= [('cls_binned',TextFile),('nl_c_array',NpzFile)]
	outputs			= [('bandpowers',TextFile),('bandpowers_MC',NpzFile)]
	
	def init_params(self):
		self.nside 		= self.config['nside']
		self.npix 		= hp.nside2npix(self.nside)
		self.pol		= self.config['pol']
		self.verbose	= self.config['verbose']
		self.Nmc		= self.config['Nsimul']
		# beam in arcmin
		self.beam      = self.config['beam']
			
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
		self.mask 	= hp.ud_grade(m,nside_out=self.nside)
		# We also need a binary mask where 0 stays 0, and >0 is 1
		self.bin_mask = np.ceil(self.mask)
		
	def read_ell_bins(self):
		bins_edges	= []
		f	= open(self.get_input('ell_bins_list'))
		for line in f:
			bins_edges.append(int(line.strip()))
		self.bins_edges	= np.array(bins_edges)
		ells	= np.arange(4*self.nside,dtype='int32')
		bpws 	= -1 + np.zeros_like(ells)
		weights = 0.25 * np.ones_like(ells)
		c		= 0
		for i in range(len(self.bins_edges)-1):
			edge_low					= self.bins_edges[i]
			edge_high					= self.bins_edges[i+1]+1
			bpws[edge_low:edge_high]	= c
			c += 1
		#for i in range(len(ells)):
		#	print(ells[i],bpws[i])
		self.bins		= nmt.NmtBin(self.nside,ells=ells,bpws=bpws,weights=weights,is_Dell=True)
		self.Nbins		= c
	
	def read_1map(self):
		# This is for reading only 1 frequency map, to calculate the coupling matrix
		f		= open(self.get_input('freq_maps_list_sp1'),'r')
		line	= f.readline()
		qu_list = hp.read_map(line.strip(), field=(1,2), verbose=False)
		self.SingleMap_s2 = nmt.NmtField(self.mask, [qu_list[0] , qu_list[1] ], purify_b=True,purify_e=True,masked_on_input=False,n_iter_mask_purify=6,n_iter=6 )
		
	def read_cmb_cls(self):
		cls_cmb			= hp.read_cl(self.get_input('cl_cmb'))
		# Fill in 
		self.nels = cls_cmb.shape[1]
		self.cmb_cls = np.zeros([4,nels])
		# EE is in index 0, BB is in index 3
		self.cmb_cls[0,:]	= cls_cmb[1]
		self.cmb_cls[3,:]	= cls_cmb[2]
	def run_decoupling_MonteCarlo(self):
		self.cls_cmb_MC = np.load(self.get_input('cl_cmb_MC'))['cls_cmb_MC']
		self.bandpowers_MC = np.zeros((self.Nmc,2,self.Nbins))
		# define the cls in the spin-2 shape
		for m in range(self.Nmc):
			cls_for_decoupling = np.zeros([4,self.nels])
			cls_for_decoupling[0,:] = self.cls_cmb_MC[m,0,:]
			cls_for_decoupling[3,:] = self.cls_cmb_MC[m,1,:]
			self.bandpowers_MC[m,:,:] = self.compute_bandpowers(cls_for_decoupling)
			print("Decoupling for iteration %i is ready"%m)
	
	def write_bandpowers_MC(self):
		fname = self.get_output('bandpowers_MC')
		np.savez(fname,bandpowers_MC=self.bandpowers_MC,Nmc=self.Nmc)

	def create_nmt_Workspace(self):
		# This just sets a nmt workspace
		self.workspace	= nmt.NmtWorkspace()
		self.workspace.compute_coupling_matrix(self.SingleMap_s2,self.SingleMap_s2,bins=self.bins,is_teb=False,n_iter=6)

	def compute_bandpowers(self,cmb_cls):
		# cmb_cls has TT,EE and BB spectra
		bandpowers	= self.workspace.decouple_cell(cmb_cls)
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

	def run(self):
		self.init_params()
		self.read_freqs()
		self.read_mask()
		
		# calculate Nell
		#self.read_ilcweights()
		#nl_c_mean_P 	= self.calculate_noise_bias(pol_nb=True)
		#nl_c_mean_T 	= self.calculate_noise_bias(pol_nb=False)
		
		# coupling matrix and binning
		self.read_ell_bins()
		self.read_1map()
		self.read_cmb_cls()
		#self.subtract_noise_bias(nl_c_mean_P)
		
		self.create_nmt_Workspace()
		bandpowers	= self.compute_bandpowers(self.cmb_cls)
		self.save_bandpowers_to_file(bandpowers,self.get_output('bandpowers'))
		
		# Now run the decoupling for the Nmc spectra
		self.run_decoupling_MonteCarlo()
		self.write_bandpowers_MC()

if __name__ == '__main__':
	cls = PipelineStage.main()
