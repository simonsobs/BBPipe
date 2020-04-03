#!/bin/bash

python_exec="/usr/bin/python3"

nside=256
sig_s=0
for seed in {1300..1304}
do
    for sig_d in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 23 25 30 40 50 60 70 80 90 100
    do
	if [ ! -f map_beta_dust_sigD${sig_d}_sd${seed}.fits ] || [ ! -f map_beta_sync_sigS${sig_s}.fits ] ; then
	    echo "Missing spectral index maps with sigma = E-2*"$sig_d
	    echo "Generating gaussian dust spectral indices with sigma "
	    python3 beta_gaussian.py --sigma-d ${sig_d} --seed ${seed}
	fi
	if [ ! -d new_sim_ns${nside}_seed${seed}_bbsim_sigD${sig_d}_sigS${sig_s} ] ; then
	    echo "Generating simulation: gaussian beta with sigma dust = "${sig_d}
	    ${python_exec} new_simulation.py --seed ${seed} --pysm-sim True --beta-dust-var True --sigma-d ${sig_d}
	fi
    done
done

## Generate simulations with pysm beta maps
#for seed in 1300 1301 1302 1303 1304
#do
#    echo "Generating simulation: pysm beta, seed = "${seed}
#    ${python_exec} new_sim.py --seed ${seed} --beta-pysm True
#done
