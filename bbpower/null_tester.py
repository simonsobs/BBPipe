from bbpipe import PipelineStage

class BBNullTester(PipelineStage):
    """
    Template for a null testing stage
    """
    name="BBNullTester"
    inputs=[]
    outputs=[]
    config_options={}

    def run(self) :
        for inp,_ in self.inputs :
            fname=self.get_input(inp)
            print("Reading "+fname)
            open(fname)

        for out,_ in self.outputs :
            fname=self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")
