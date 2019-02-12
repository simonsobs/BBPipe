import numpy as np

from bbpipe import PipelineStage
from .types import DummyFile

import corner 

class PlottingCompSep(PipelineStage):
    """
    This stage hopefully plots cool things. 
    """
    name = "PlottingCompSep"
    inputs = [('samples', DummyFile)]
    outputs = [('plots', DummyFile)]
    
    def run(self):
        self.plot_things_yay()

    def plot_things_yay(self):
        fig = corner.corner(samples, plot_datapoints=False, bins=100, levels=[0.68, 0.95], smooth=1., \
                            labels=names, label_kwargs=dict(fontsize=20))
        savefig('happyname.pdf', dpi=300, format='pdf')


if __name__ == '__main__':
    sample_plots = PipelineStage.main()
