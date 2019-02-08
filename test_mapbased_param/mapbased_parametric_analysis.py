from bbpipe import PipelineStage
from .types import FitsFile
import numpy as np
import pylab as pl

class BBMapParamCompSep(PipelineStage):
    """
    Stage that performs three things:
        * fit of the spectral indices, using fgbuster tools, along with their error bars
        * estimate sky components, foregrounds and CMB
        * estimate components' covariance
    """
    name='BBMapParamCompSep'
    inputs= [('binary_mask',FitsFile),('frequency_maps',FitsFile),('noise_cov',FitsFile)]
    outputs=[('post_compsep_maps',FitsFile)]

    def run(self) :
        #Read input mask
        import healpy as hp #We will want to be more general than assuming HEALPix
        binary_mask=hp.read_map(self.get_input('binary_mask'),verbose=False)

        #Read frequency maps and noise covariance
        frequency_maps=hp.read_map(self.get_input('frequency_maps'),verbose=False, field=None)
        noise_cov=hp.read_map(self.get_input('noise_cov'),verbose=False, field=None)

        # perform component separation
        # assuming inhomogeneous noise
        import fgbuster as fg
        from fgbuster.component_model import CMB, Dust, Synchrotron
        components = [CMB(), Dust(150.), Synchrotron(150.)]
        instrument = {'frequencies'=[30.0, 40.0, 95.0, 150.0, 220.0, 270.0]}

        from fgbuster.separation_recipies import weighted_comp_sep
        res = fg.separation_recipies.weighted_comp_sep(components, instrument,
                     data=frequency_maps, cov=noise_cov, nside=0)

        # save results
        # fits for components maps
        # and text file for spectral parameters with error bars?
        # or maps for everything maybe....
        results_maps = np.hstack((res.s, res.invAtNA, res.x, res.Sigma.diag()))
        hp.write_map(self.get_output('post_compsep_maps'), results_maps,overwrite=True)

if __name__ == '__main__':
    results = PipelineStage.main()
