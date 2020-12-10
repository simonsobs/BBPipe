import numpy as np
from bbpipe import PipelineStage
from .types import NpzFile, TextFile


class Likelihood(PipelineStage):
    name = "Likelihood"
    inputs = [('c_ell', NpzFile), ('cl_input', TextFile)]
    outputs = [('results', NpzFile)]
    config_options = {'nside': 16, 'lmax': 32, 'lmin': 10}

    def set_up(self):
        _, cl_in = np.loadtxt(self.get_input('cl_input'), unpack=True)
        d = np.load(self.get_input('c_ell'))
        self.ls = d['ls']
        self.cl = d['cl']
        self.tl = cl_in[self.ls]
        self.icov = d['cov']

    def run(self):
        s2_a = 1./np.dot(self.tl, np.dot(self.icov, self.tl))
        a = s2_a * np.dot(self.tl, np.dot(self.icov, self.cl))
        np.savez(self.get_output('results'), a=a, sigma_a=np.sqrt(s2_a))


if __name__ == '__main__':
    cls = PipelineStage.main()
