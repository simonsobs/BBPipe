from bbpipe import PipelineStage
from .types import DummyFile
from sacc.sacc import SACC
import numpy as np

class BBCompSep(PipelineStage):
    """
    Template for a component separation stage
    """
    name="BBCompSep"
    inputs=[('sacc_file',SACC)]
    outputs=[('param_chains',DummyFile)]
    config_options={'foreground_model':'none'}

    def parse_sacc_file(self) :
        #Read sacc file
        self.s=SACC.loadFromHDF(self.get_input('sacc_file'))
        #This just prints information about the contents of this file
        self.s.printInfo()

        #Get power spectrum ordering
        self.order=self.s.sortTracers()

        #Get bandpasses
        self.bpasses=[[t.z,t.Nz] for t in self.s.tracers]

        #Get bandpowers
        self.bpw_l=self.s.binning.windows[0].ls
        # We're assuming that all bandpowers are sampled at the same values of ell.
        # This is the case for the BICEP data and we may enforce it, but it is not
        # enforced within SACC.
        self.bpw_w=np.array([w.w for w in self.s.binning.windows])
        # At this point self.bpw_w is an array of shape [n_bpws,n_ells], where 
        # n_bpws is the number of power spectra stored in this file.

        #Get data vector
        self.data=self.s.mean.vector

        #Get covariance matrix
        self.covar=self.s.precision.getCovarianceMatrix()
        # TODO: At this point we haven't implemented any scale cuts, or cuts
        # on e.g. using only BB etc.
        # This could be done here with some SACC routines if needed.

    def run(self) :
        # First, read SACC file containing reduced power spectra,
        # bandpasses, bandpowers and covariances.
        self.parse_sacc_file()
            
        #This stage currently does nothing whatsoever

        #Write outputs
        for out,_ in self.outputs :
            fname=self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")

if __name__ == '__main__':
    cls = PipelineStage.main()
