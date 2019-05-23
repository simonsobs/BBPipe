from bbpipe import PipelineStage
from .types import FitsFile,TextFile,SACCFile
from sacc.sacc import SACC
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


class BBPowerSpecter(PipelineStage):
    """
    Template for a power spectrum stage
    """
    name="BBPowerSpecter"
    inputs=[('splits_list',TextFile),('masks_apodized',FitsFile),('bandpasses_list',TextFile),
            ('sims_list',TextFile),('beams_list',TextFile)]
    outputs=[('cells_all_splits',SACCFile),('cells_all_sims',TextFile)]
    config_options={'bpw_edges':None,
                    'beam_correct':True,
                    'purify_B':True}

    def init_params(self):
        self.nside = self.config['nside']

    def read_beams(self,nbeams):
        from scipy.interpolate import interp1d

        beam_fnames = []
        with open(self.get_input('beams_list'),'r') as f:
            for fname in f:
                beam_fnames.append(fname.strip())

        # Check that there are enough beams
        if len(beam_fnames)!=nbeams:
            raise ValueError("Couldn't find enough beams %d != %d" (len(beam_fnames),nbeams))

        self.larr_all = np.arange(3*self.nside)
        self.beams={}
        for i_f,f in enumerate(beam_fnames):
            li,bi=np.loadtxt(f,unpack=True)
            bb=interp1d(li,bi,fill_value=0,bounds_error=False)(self.larr_all)
            if li[0]!=0:
                bb[:int(li[0])]=bi[0]
            self.beams['band%d' % (i_f+1)]=bb

    def read_bandpasses(self):
        bpss_fnames = []
        with open(self.get_input('bandpasses_list'),'r') as f:
            for fname in f:
                bpss_fnames.append(fname.strip())
        self.n_bpss = len(bpss_fnames)
        self.bpss={}
        for i_f,f in enumerate(bpss_fnames):
            nu,bnu=np.loadtxt(f,unpack=True)
            dnu=np.zeros_like(nu)
            dnu[1:]=np.diff(nu)
            dnu[0]=dnu[1]
            self.bpss['band%d' % (i_f+1)]={'nu':nu, 'dnu':dnu, 'bnu':bnu}

    def read_masks(self,nbands):
        self.masks=[]
        for i in range(nbands):
            m=hp.read_map(self.get_input('masks_apodized'),
                          verbose=False)
            self.masks.append(hp.ud_grade(m,nside_out=self.nside))

    def run(self) :
        self.init_params()

        # Read bandpasses
        self.read_bandpasses()

        # Read beams
        self.read_beams(self.n_bpss)

        # Read masks
        self.read_masks(self.n_bpss)

        # Compute all possible MCMs
        
        # Compile list of splits
        # Compute all possible cross-power spectra
        # Iterate over simulations
        #   Compute list of splits
        #   Compute all possible cross-power spectra
        # Save output
        #   Save cells_all_splits
        #   Iterate over simulations
        #     Save cell for this sim, and add name to list
        #   Write sims list to cells_all_sims
        print("HI")

if __name__ == '__main__':
    cls = PipelineStage.main()
