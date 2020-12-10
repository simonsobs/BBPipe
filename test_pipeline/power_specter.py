import numpy as np
import healpy as hp
from bbpipe import PipelineStage
from .types import FitsFile, NpzFile


class PowerSpecter(PipelineStage):
    name = "PowerSpecter"
    inputs = [('map', FitsFile)]
    outputs = [('c_ell', NpzFile)]
    config_options = {'nside': 16, 'lmax': 32, 'lmin': 10}

    def run(self):
        m = hp.read_map(self.get_input('map'), verbose=False)
        ls = np.arange(3*self.config['nside'])
        cl = hp.anafast(m)
        msk = (ls < self.config['lmax']) & (ls > self.config['lmin'])
        ls = ls[msk]
        cl = cl[msk]
        cov = np.diag(cl**2/(ls+0.5))

        np.savez(self.get_output('c_ell'), ls=ls, cl=cl, cov=cov)


if __name__ == '__main__':
    cls = PipelineStage.main()
