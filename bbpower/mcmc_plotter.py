import numpy as np
import os
from matplotlib import pyplot
import emcee
import corner

from bbpipe import PipelineStage
from .types import NpzFile, HDFFile

class BBMcmcPlotter(PipelineStage):
    name = "BBMcmcPlotter"
    inputs = [('sampler_out', HDFFile), ('paramnames', NpzFile)]
    
    def clean_samples(self):

    def make_plots(self):

    
    def run(self):
        mcsamples = self.clean_samples()
        self.make_plots()
        
        return 



if __name__ == '__main__':
    PipelineStage.main()
