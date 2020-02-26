
N_mc 	= 100
f		= open('noise_simulations_files.dat','w')

for n in range(N_mc):
	f.write('/Users/chervias/Documents/postdoc_FSU/SO-BB/data/Noise_MC_simulations/NoiseMap_n%i_ns512_tqu_freq30.0.fits\n'%n)
	f.write('/Users/chervias/Documents/postdoc_FSU/SO-BB/data/Noise_MC_simulations/NoiseMap_n%i_ns512_tqu_freq40.0.fits\n'%n)
	f.write('/Users/chervias/Documents/postdoc_FSU/SO-BB/data/Noise_MC_simulations/NoiseMap_n%i_ns512_tqu_freq90.0.fits\n'%n)
	f.write('/Users/chervias/Documents/postdoc_FSU/SO-BB/data/Noise_MC_simulations/NoiseMap_n%i_ns512_tqu_freq150.0.fits\n'%n)
	f.write('/Users/chervias/Documents/postdoc_FSU/SO-BB/data/Noise_MC_simulations/NoiseMap_n%i_ns512_tqu_freq220.0.fits\n'%n)
	f.write('/Users/chervias/Documents/postdoc_FSU/SO-BB/data/Noise_MC_simulations/NoiseMap_n%i_ns512_tqu_freq270.0.fits\n'%n)

f.close()
