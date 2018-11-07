from bbpipe import PipelineStage
from .types import FitsFile,YamlFile,DummyFile

class BBCompSep(PipelineStage):
    """
    Template for a component separation stage
    """
    name="BBCompSep"
    inputs=[('splits_info',YamlFile),('power_spectra_splits',DummyFile),
            ('covariance_matrix',DummyFile)]
    outputs=[('param_chains',DummyFile)]
    config_options={}

    def run(self) :
        #This stage currently does nothing whatsoever
        print(self.config)
        for inp,_ in self.inputs :
            fname=self.get_input(inp)
            print("Reading "+fname)
            open(fname)

        for out,_ in self.outputs :
            fname=self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")

if __name__ == '__main__':
    cls = PipelineStage.main()
