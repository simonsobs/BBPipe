from bbpipe import PipelineStage
from .types import FitsFile, YamlFile, TextFile, NmtFieldFile

class MapPreproc(PipelineStage):
    """
    Template for a map pre-processing stage
    """
    name="MapPreproc"
    inputs=[('raw_splits',FitsFile),('window_function',FitsFile)]
    outputs=[]#('nmt_fields',NmtFieldFile)]
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

class MaskPreproc(PipelineStage):
    """
    Template for a map pre-processing stage
    """
    name='MaskPreproc'
    inputs= [('binary_mask',FitsFile),('source_data',TextFile)]
    outputs=[('window_function',FitsFile)]
    config_options={'aposize_edges':1.0,
                    'apotype_edges':'C1',
                    'aposize_srcs':0.1,
                    'apotype_srcs':'C1'}

    def run(self) :
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
