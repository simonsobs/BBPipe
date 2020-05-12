#!/bin/bash

mkdir -p test/test_out

# Generate fiducial cls
python ./examples/generate_SO_spectra.py test/test_out

# Generate simulations
for seed in {1001..1010}
do
    mkdir -p test/test_out/s${seed}
    echo ${seed}
    python examples/generate_SO_maps.py --output-dir test/test_out/s${seed} --seed ${seed} --nside 64
done

# Run pipeline
python -m bbpower BBPowerSpecter   --splits_list=./examples/test_data/splits_list.txt   --masks_apodized=./examples/test_data/masks_ones.fits.gz   --bandpasses_list=./examples/data/bpass_list.txt   --sims_list=./examples/test_data/sims_list.txt   --beams_list=./examples/data/beams_list.txt   --cells_all_splits=./test/test_out/cells_all_splits.fits   --cells_all_sims=./test/test_out/cells_all_sims.txt   --mcm=./test/test_out/mcm.dum   --config=./test/test_config.yml

# Cleanup
rm -r test/test_out
