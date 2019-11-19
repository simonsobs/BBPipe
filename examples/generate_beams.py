from noise_calc import Simons_Observatory_V3_SA_noise,Simons_Observatory_V3_SA_beams
import numpy as np
import matplotlib.pyplot as plt

names=['LF1','LF2','MF1','MF2','UHF1','UHF2']

beams=Simons_Observatory_V3_SA_beams()

larr=np.arange(10000)
for ib,b in enumerate(beams):
    sigma = b *np.pi/180./60./2.355
    bb=np.exp(-0.5*sigma**2*larr*(larr+1))
    np.savetxt("beam_"+names[ib]+'.txt',np.transpose([larr,bb]))
    plt.plot(larr,bb)
plt.show()
