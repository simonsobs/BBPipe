#!/bin/bash

# both sync and dust varying
for seed in {1001..1010}
do
    for std in 20 
    do
	echo "Creating output directory if not existent"
	[ -d /mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std} ] || mkdir /mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}

	echo "Creating single simulation list if not existent"
	[ -f test_BBMoments_simulations/list_sim_sd${seed}_stdd${std}_stds${std}.txt ] ||
	echo "/mnt/extraspace/susanna/BBMoments/Simulations_Moments_varStd/sim_ns256_seed${seed}_stdd${std}_stds${std}_gdm3.0_gsm3.0_msk_B" >> list_sim_sd${seed}_stdd${std}_stds${std}.txt
	
	echo "BBPSPec for seed = "${seed}", std_s,std_d = "${std}
	addqueue -s -q berg -m 2 -n 1x12 /usr/bin/python3 -m bbpower BBPowerSpecter   --splits_list=/mnt/extraspace/susanna/BBMoments/Simulations_Moments_varStd/sim_ns256_seed${seed}_stdd${std}_stds${std}_gdm3.0_gsm3.0_msk_B/splits_list.txt   --masks_apodized=./test_BBMoments_simulations/masks_SAT_ns256.fits   --bandpasses_list=./test_BBMoments_simulations/bandpasses.txt   --sims_list=./test_BBMoments_simulations/list_sim_sd${seed}_stdd${std}_stds${std}.txt   --beams_list=./test_BBMoments_simulations/beams.txt   --cells_all_splits=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_all_splits.sacc   --cells_all_sims=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_all_sims.txt   --mcm=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/mcm.dum   --config=./test_BBMoments_simulations/config.yaml
    done
done


