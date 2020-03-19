#!/bin/bash

python_exec="/usr/bin/python3"

#if [ ! -f map_beta_dust.fits ] || [ ! -f map_beta_sync.fits ] ; then
#    echo "Missing gaussian beta maps with unit amplitude"
#fi
#
#
#echo "Generating gaussian spectral indices maps"
#python3 beta_gaussian.py
#
#
#nside=256
#for seed in {1300..1304}
#do
#    if [ ! -d new_sim_ns${nside}_seed${seed}_bbsim_betaVar ] ; then
#	echo "Generate simulation with gaussian spectral index"
#	addqueue -q berg -m 32 ${python_exec} new_sim.py --seed ${seed} --beta-var True
#	echo "/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/new_sim_ns${nside}_seed${seed}_bbsim" >>  list_newsims_GausBetas_fullsky_noinst.txt
#    fi
#done


nside=256
seed=1300
Async=1
for Adust in {50..500..50}
do
    if [ ! -f map_beta_dust_Ad${Adust}.fits ] || [ ! -f map_beta_sync_As${Async}.fits ] ; then
	echo "Missing spectral index maps with amplitude = "$Adust
	echo "Generating gaussian dust spectral indices with amplitude = "$Adust
	python3 beta_gaussian.py --A-bd ${Adust}
    fi
    if  [ ! -d new_sim_ns${nside}_seed${seed}_bbsim_betaVar_Ad${Adust}As${Async}_bbsim ] ; then
	echo "Generating simulation: gaussian beta with Adust = "$Adust
	${python_exec} new_sim.py --seed ${seed} --beta-var True --A-bd ${Adust}
	echo "/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/new_sim_ns${nside}_seed${seed}_bbsim_betaVar_Ad${Adust}As${Async}" >>  list_newsims_seed1300_AdAs.txt
    fi
done

#addqueue -q berg -m 22

	     

