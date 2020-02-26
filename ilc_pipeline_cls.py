import numpy as np
import healpy as hp
import pymaster as nmt
from bbpipe import PipelineStage
from types_mine import FitsFile,TextFile,DummyFile,NpzFile

class BB_Cl_Process(PipelineStage):
	name 			= 'BB_Cl_Process'
	inputs			= [('freq_maps_list',TextFile),('cl_cmb',FitsFile),('ilc_weights',NpzFile),('ell_bins_list',TextFile),('freqs_list',TextFile),('masks',TextFile),('noise_sims_list',TextFile)]
	outputs			= [('cls_binned',TextFile),('nl_c_array',NpzFile)]
	
	def init_params(self):
		self.nside 		= self.config['nside']
		self.npix 		= hp.nside2npix(self.nside)
		self.pol		= self.config['pol']
		self.verbose	= self.config['verbose']
		self.Nsimul		= self.config['Nsimul']
			
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
	
	def read_1map(self):
		# This is for reading only 1 frequency map, to calculate the coupling matrix
		f		= open(self.get_input('freq_maps_list'),'r')
		line	= f.readline()
		self.SingleMap_s0 = nmt.NmtField(self.mask, [hp.read_map(line.strip(), field=0, verbose=False)])
		self.SingleMap_s2 = nmt.NmtField(self.mask, hp.read_map(line.strip(), field=(1,2), verbose=False))
		
	def read_cmb_cls(self):
		# Fill in for 7 cls
		cmb_cls_teb			= hp.read_cl(self.get_input('cl_cmb'))
		self.cmb_cls		= np.zeros((7,cmb_cls_teb.shape[1]))
		self.cmb_cls[0:3,:]	= cmb_cls_teb
	
	def read_ilcweights(self):
		npzfile 		= np.load(self.get_input('ilc_weights'))
		self.ilc_w_T 	= npzfile['ilc_weights_T']
		self.ilc_w_P	= npzfile['ilc_weights_P']
	
	def create_nmt_Workspace(self):
		# This just sets a nmt workspace
		self.workspace	= nmt.NmtWorkspace()
		self.workspace.compute_coupling_matrix(self.SingleMap_s0,self.SingleMap_s2,bins=self.bins,is_teb=True)
	
	def compute_binned_cl(self):
		# cmb_cls has TT,EE and BB spectra
		cmb_cls_binned	= self.workspace.decouple_cell(self.cmb_cls)
		return cmb_cls_binned
	
	def save_binned_cls_to_file(self,cl,fname):
		print('Saving to file ...'+fname)
		f		= open(fname,'w')
		Nbin	= self.bins.get_n_bands()
		for i in range(Nbin):
			ells_in_bin	= self.bins.get_ell_list(i)
			ell_low		= ells_in_bin[0]
			ell_high	= ells_in_bin[-1]
			f.write('%i\t%i\t%E\t%E\t%E\t%E\t%E\t%E\t%E\n'%(ell_low,ell_high,cl[0,i],cl[1,i],cl[2,i],cl[3,i],cl[4,i],cl[5,i],cl[6,i]))
		f.close()
	def get_noise_sims(self):
		f	= open(self.get_input('noise_sims_list'),'r')
		noise_sims_array = []
		for n in range(self.Nsimul):
			noise_sims_arr_n	= []
			for line in f:
				noise_tqumap 	= hp.read_map(line.replace('\n',''),field=(0,1,2),nest=False)
				noise_sims_arr_n.append(noise_tqumap)
			noise_sims_array.append(noise_sims_arr_n)
		return noise_sims_array
	
	def calculate_noise_bias(self,noise_sims_maps):
		# noise_sims_maps should have a size N_sims, inside there should be an array of size Nfreqs, and inside an array of size 3 with T,Q,U map
		# this will receive the noise simulations and calculate a noise bias
		def comp_alm_nmt(mp,pol=False):
			if pol :
				return nmt.NmtField(self.mask,[mp[1],mp[2]],purify_b=False)
			else :
				return nmt.NmtField(self.mask,[mp])
		def comp_pspec_nmt(alm1,alm2) :
			cls=nmt.compute_coupled_cell(alm1,alm2)
			if len(alm1.get_maps())==2 :
				return np.array([cls[0],cls[3],cls[1]])
			else :
				return cls		
		
		n_ells		= np.zeros([self.Nsimul,2,self.Nfreqs,self.Nfreqs,3*self.nside]) #Noise power spectrum (fg-full)
		nl_c_array	= np.zeros((self.Nsimul,2,3*self.nside))
		for i in range(self.Nsimul):
			# run Nsimul cases
			alms	= np.array([comp_alm_nmt(n,pol=True) for n in noise_sims_maps[i]])
			for inu1 in range(self.Nfreqs) :
				for inu2 in range(inu1,self.Nfreqs):
					cls=comp_pspec_nmt(alms[inu1],alms[inu2])
					for ip in np.arange(2) :
						n_ells[i,ip,inu1,inu2,:]=cls[ip]
						if inu1!=inu2 :
							n_ells[i,ip,inu2,inu1,:]=cls[ip]
			nl_c=np.sum(self.ilc_w_P*np.sum(n_ells[i,:,:,:,:]*self.ilc_w_P[:,None,:,:],axis=2),axis=1)
			nl_c_array[i,:,:] = nl_c
		# Now we calculate the average noise bias along the Nsimul simulations
		# along the 
		nl_c_mean = np.mean(nl_c_array,axis=0)
		return nl_c_mean
	
	def subtract_noise_bias(self,noise_bias):
		# this will subtract the noise bias to the cl estimated by HILC, the noise is calculated ONLY for polarization an has shape (2,Nell) , with the 2 corresponding to EE and BB
		# this function will modify self.cmb_cls in place
		# the cl_c has shape (7,Nell) because the Workspace needs the 7 cls. 0 is TT, 1 is EE, 2 is BB, the rest are filled with zeros
		for i in [1,2]:
			# the i -1 is because 0,1 in noise bias corresponds to 1,2 in cl_c
			self.cmb_cls[i,:] = self.cmb_cls[i,:] - noise_bias[i-1,:]	
	def save_to_npz(self,fname,nl_c_array):
		np.savez(fname,nl_c_array=nl_c_array)

	def run(self):
		self.init_params()
		self.read_freqs()
		self.read_mask()
		
		# calculate Nell
		self.read_ilcweights()
		noise_sims_array 	= self.get_noise_sims()
		nl_c_mean 			= self.calculate_noise_bias(noise_sims_array)
		#self.save_to_npz(self.get_output('nl_c_array'),nl_c_array)
		
		# coupling matrix and binning
		self.read_ell_bins()
		self.read_1map()
		self.read_cmb_cls()
		self.subtract_noise_bias(nl_c_mean)
		
		self.create_nmt_Workspace()
		cls_binned	= self.compute_binned_cl()
		self.save_binned_cls_to_file(cls_binned,self.get_output('cls_binned'))

if __name__ == '__main__':
	cls = PipelineStage.main()
