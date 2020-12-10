import numpy as np


def test_pipeline_output():
    d = np.load('test/results.npz')
    # Less than a 5-sigma deviation
    assert np.fabs(d['a']-1)/d['sigma_a'] < 5
