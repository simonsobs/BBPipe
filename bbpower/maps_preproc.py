from bbpipe import PipelineStage
from .types import FitsFile, NmtFieldFile

class BBMapsPreproc(PipelineStage):
    """
    Template for a map pre-processing stage
    """
    name="BBMapsPreproc"
    inputs=[('raw_splits',FitsFile),('window_function',FitsFile)]
    outputs=[('nmt_fields',NmtFieldFile)]
    config_options={'purify_b':False}

    def run(self) :
        for inp,_ in self.inputs :
            fname=self.get_input(inp)
            print("Reading "+fname)
            open(fname)

        for out,_ in self.outputs :
            fname=self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")
