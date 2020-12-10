import numpy as np
import healpy as hp
from .types import TextFile, FitsFile
from bbpipe import PipelineStage


class MapGenerator(PipelineStage):
    name = "MapGenerator"
    inputs = [('cl_input', TextFile)]
    outputs = [('map', FitsFile)]
    config_options = {'nside': 16}

    def run(self):
        l, cl = np.loadtxt(self.get_input('cl_input'), unpack=True)
        m = hp.synfast(cl, self.config['nside'], verbose=False)
        hp.write_map(self.get_output('map'), m, overwrite=True)


if __name__ == '__main__':
    cls = PipelineStage.main()
