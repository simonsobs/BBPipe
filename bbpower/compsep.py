from bbpipe import PipelineStage
from .types import DummyFile
from sacc.sacc import SACC

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
        

    def run(self) :
        # First, read SACC file containing reduced power spectra,
        # bandpasses, bandpowers and covariances.
        self.parse_sacc_file()
        #This stage currently does nothing whatsoever

        for out,_ in self.outputs :
            fname=self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")

if __name__ == '__main__':
    cls = PipelineStage.main()
