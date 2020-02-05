import numpy as np
import emcee
import corner
from matplotlib import pyplot as plt

fdir = '/mnt/zfsusers//mabitbol/BBPipe/updated_runs/combined_bandsangles/output2/'
fsamps = np.load(fdir+'cleaned_chains.npz')
samples = fsamps['samples']
labels = fsamps['labels']

fig = corner.corner(samples, labels=labels, bins=100);
plt.savefig('/mnt/zfsusers//mabitbol/notebooks/data_and_figures/combined_bandsangles_full_triangle.png', fmt='png')
