#!/bin/bash

python_exec="/usr/bin/python3"

if [ ! -f map_beta_dust.fits ] || [ ! -f map_beta_sync.fits ] ; then
    echo "Missing gaussian beta maps"

fi

echo "Generating gaussian spectral indices maps"
python3 beta_gaussian.py

nside=256
for seed in {1300..1304}
do
    if [ ! -d new_sim_ns${nside}_seed${seed}_bbsim_betaVar ] ; then
	echo "Generate simulation with gaussian spectral index"
	addqueue -q berg -m 32 ${python_exec} new_sim.py --seed ${seed} --beta-var True
	echo "/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/new_sim_ns${nside}_seed${seed}_bbsim" >>  list_newsims_GausBetas_fullsky_noinst.txt
    fi
done


nside=256
seed=1300
Async=2
for Adust in {5..50..5}
do
    if  [ ! -d new_sim_ns${nside}_seed${seed}_Ad${Adust}_As${Async}_bbsim ] ; then
	echo "Genering simulation with dust amplitude ="$Adust
	addqueue -q berg -m 32 ${python_exec} new_sim.py --A-dust ${Adust}
	echo "/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/new_sim_ns${nside}_seed${seed}_Ad${Adust}_As${Async}_bbsim" >>  list_newsims_Adust_consBetas_fullsky_noinst.txt
    fi
done
	     

