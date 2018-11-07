from bbpipe import PipelineStage
from .types import FitsFile, TextFile
import numpy as np

class BBMaskPreproc(PipelineStage):
    """
    Template for a mask pre-processing stage
    """
    #The following 3 attributes are required for any PipelineStage
    #Name. Make it coincide with the class name for convenience.
    name='BBMaskPreproc'
    #Inputs. List of tuples.
    # Each tuple contains the name of the dataset and the class describing its format.
    #   See types.py for a few examples of format classes
    # The name of the dataset is either the key used in a yaml config file or the key
    #   assigned to it by a preceding pipeline stage (e.g. the name of an output)
    # Note that you can access the file names associated with each input by using
    #   the method PipelineStage.get_input(input_name).
    inputs= [('binary_mask',FitsFile),('source_data',TextFile)]
    #Outputs. List of tuples.
    # Each tuple contains the same information as the `inputs` tuples.
    # The name will be used by subsequent pipline stages in their inputs.
    # Note that BBPipe assigns file names to each output based on their name, on the
    #   contents of the configuration files and on their data type. To retrieve the
    #   file names use the method PipelineStage.get_output(output_name)
    outputs=[('window_function',FitsFile)]
    #You can also define default configuration options. These get overwritten by any
    #options set in this stage's configuration.
    config_options={'aposize_edges':1.0,
                    'apotype_edges':'C1',
                    'aposize_srcs':0.1,
                    'apotype_srcs':'C1'}

    def run(self) :
        #Read input mask
        import healpy as hp #We will want to be more general than assuming HEALPix
        mask_raw=hp.read_map(self.get_input('binary_mask'),verbose=False)

        #Read point source data
        # Right now this is a simple text file, but this is probably not ideal.
        ps_ra,ps_dec,ps_size=np.loadtxt(self.get_input('source_data'),unpack=True,ndmin=2)

        #Now we should do stuff (apodization, inverse-variance weighting etc.)
        #but this is just a placeholder
        print("BBMaskPreproc currently does nothing")

        #Write window function
        #Currently just writing the input mask.
        hp.write_map(self.get_output('window_function'),mask_raw,overwrite=True)

if __name__ == '__main__':
    cls = PipelineStage.main()
