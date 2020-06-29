import numpy as np
import healpy as hp
import pymaster as nmt
from bbpipe import PipelineStage
from types_mine import FitsFile,TextFile,DummyFile,NpzFile

class BB_HILC(PipelineStage):
	name 			= 'BB_HILC'
	inputs			= [('freq_maps_list',TextFile),('ell_bins_list',TextFile),('freqs_list',TextFile),('masks',TextFile)]
	outputs			= [('cl_cmb',FitsFile),('ilc_weights',NpzFile),('bands',NpzFile)]
	
	def init_params(self):
		self.nside 		= self.config['nside']
		self.npix 		= hp.nside2npix(self.nside)
		self.verbose	= self.config['verbose']
			
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

	def read_mask(self):
		m			= hp.read_map(self.get_input('masks'),verbose=False)
		self.mask 	= hp.ud_grade(m,nside_out=self.nside)
		#print(hp.npix2nside(len(self.mask)))
	
	def read_ell_bins(self):
		bins_edges	= []
		f	= open(self.get_input('ell_bins_list'))
		for line in f:
			bins_edges.append(int(line.strip()))
		self.bins_edges	= np.array(bins_edges)
	
	def read_freq_maps(self):
		# This is for reading the maps files and return it as a list of namaster fields
		f	= open(self.get_input('freq_maps_list'),'r')
		self.maps_s2 = [] 
		self.maps_s0 = []
		for line in f:
			self.maps_s0.append(nmt.NmtField(self.mask, [hp.read_map(line.strip(), field=0, verbose=False)]))
			self.maps_s2.append(nmt.NmtField(self.mask, hp.read_map(line.strip(), field=(1,2), verbose=False)))
	
	def clean_ilc(self,alms_list,pol):
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
		
		def cmb(nu):
			x = 0.0176086761 * nu
			ex = np.exp(x)
			sed = ex * (x / (ex - 1)) ** 2
			return sed
		
		n_nu=len(freqs)
		c_nu=cmb(freqs) #Frequency dependence
		if len(alms_list)!=len(c_nu) :
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
				cls=pspec_compute(alms_list[inu1],alms_list[inu2])
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
		self.init_params()
		self.read_freqs()
		self.read_mask()
		
		self.read_ell_bins()
		self.read_freq_maps()
		
		resultT		= self.clean_ilc(self.maps_s0,pol=False)
		resultP 	= self.clean_ilc(self.maps_s2,pol=True)
		
		np.savez(self.get_output('bands'),bands = resultP['bands'])
		
		# join the spectra
		cls_cmb			= [resultT['cl_c'][0],resultP['cl_c'][0],resultP['cl_c'][1]]
		# save cell cmb
		self.save_cl_to_file(cls_cmb,self.get_output('cl_cmb'))
		# save ilc weights
		self.save_ilcweights_to_file(resultT['w_ilc'],resultP['w_ilc'],self.get_output('ilc_weights'))

if __name__ == '__main__':
	cls = PipelineStage.main()
