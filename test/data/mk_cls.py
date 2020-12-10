import numpy as np

nside = 256
ls = np.arange(3*nside)
cls = 1./(ls+10)
np.savetxt('cls.txt', np.transpose([ls, cls]), fmt='%d %lE')
