#!/bin/bash

python_exec="/usr/bin/python3"

if [ ! -f map_beta_dust.fits ] ; then
    echo "Missing dust beta map"
    if [ ! -f map_beta_sync.fits ] ; then
	echo "Missing dust beta map"
fi

echo "Generating gaussian spectral indices"
python3 beta_gaussian.py

fi

for seed in {1300..1301}
do
    ${python_exec} new_sim.py --seed ${seed} --beta-var True
    echo "/mnt/extraspace/susanna/SO/new_sims_constBetas/new_sim_ns${nside}_seed${seed}_bbsim" >> list_newsims_GausBetas_fullsky_noinst.txt
done
