#!/bin/bash

# --------------------------------------
# BBPSpec: compute cells for new single sim
# both sync and dust varying
# --------------------------------------
for seed in {1000..1010}
do
    for std in 30 #20 #50
    do
	echo "Creating output directory if not existent"
	[ -d /mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std} ] ||
	    mkdir /mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}

	echo "Creating single simulation list if not existent"
	[ -f ./test_BBMoments_simulations/list_sim_sd${seed}_stdd${std}_stds${std}.txt ] ||
	    echo "/mnt/extraspace/susanna/BBMoments/Simulations_Moments_varStd/sim_ns256_seed${seed}_stdd${std}_stds${std}_gdm3.0_gsm3.0_msk_B" >> ./test_BBMoments_simulations/list_sim_sd${seed}_stdd${std}_stds${std}.txt
	
	echo "BBPSPec for seed = "${seed}", std_s,std_d = "${std}
	addqueue -s -q berg -m 2 -n 1x12 /usr/bin/python3 -m bbpower BBPowerSpecter   --splits_list=/mnt/extraspace/susanna/BBMoments/Simulations_Moments_varStd/sim_ns256_seed${seed}_stdd${std}_stds${std}_gdm3.0_gsm3.0_msk_B/splits_list.txt   --masks_apodized=./test_BBMoments_simulations/masks_SAT_ns256.fits   --bandpasses_list=./test_BBMoments_simulations/bandpasses.txt   --sims_list=./test_BBMoments_simulations/list_sim_sd${seed}_stdd${std}_stds${std}.txt   --beams_list=./test_BBMoments_simulations/beams.txt   --cells_all_splits=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_all_splits.sacc   --cells_all_sims=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_all_sims.txt   --mcm=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/mcm.dum   --config=./test_BBMoments_simulations/config.yaml
    done
done

# --------------------------------------
# Rename computed cells_all_splits_sim0.sacc to prevent overwriting
# Copy already computed cells_all_splits_simX.sacc to directory with previously computed one
# Rename reverse
# Remove cells_all_sims.txt with single element and add all 500 elements to txt
# --------------------------------------
for seed in {1000..1010}
do
    for std in 30 
    do
	out_dir=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/
	echo "Before running pipeline prepare output directory "${out_dir}
	rm ${out_dir}cells_all_splits.sacc
	rm ${out_dir}*.dat
	mv ${out_dir}cells_all_splits_sim0.sacc ${out_dir}cells_all_splits_sim0NEW.sacc
	cp /mnt/extraspace/susanna/BBMoments/Output_BBMoments_500sims_std0_msk_12cores_onlyB/cells_all_splits_sim*.sacc ${out_dir}
	mv ${out_dir}cells_all_splits_sim0NEW.sacc ${out_dir}cells_all_splits_sim0.sacc
	rm ${out_dir}cells_all_sims.txt 

	for nsim in {0..499}
	do
	    echo "${out_dir}cells_all_splits_sim${nsim}.sacc" >> ${out_dir}cells_all_sims.txt
	done
    done
done

# --------------------------------------
# BBPSpec with 500 sims
# both sync and dust varying
# --------------------------------------
for seed in {1000..1010}
do
    for std in 30
    do
	echo "BBPSPec for seed = "${seed}", std_s,std_d = "${std}
	addqueue -s -q berg -m 2 -n 1x12 /usr/bin/python3 -m bbpower BBPowerSpecter   --splits_list=/mnt/extraspace/susanna/BBMoments/Simulations_Moments_varStd/sim_ns256_seed${seed}_stdd${std}_stds${std}_gdm3.0_gsm3.0_msk_B/splits_list.txt   --masks_apodized=./test_BBMoments_simulations/masks_SAT_ns256.fits   --bandpasses_list=./test_BBMoments_simulations/bandpasses.txt   --sims_list=./test_BBMoments_simulations/list_sims_sd${seed}_std${std}_masked_COMPLETE.txt   --beams_list=./test_BBMoments_simulations/beams.txt   --cells_all_splits=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_all_splits.sacc   --cells_all_sims=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_all_sims.txt   --mcm=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/mcm.dum   --config=./test_BBMoments_simulations/config.yaml
    done
done

# --------------------------------------
# BBPSummarizer with 500 sims
# both sync and dust varying
# --------------------------------------
for seed in {1000..1010}
do
    for std in 30
    do
	echo "BBPSummarizer for seed = "${seed}", std = "${std}
	addqueue -q berg -m 10 /usr/bin/python3 -m bbpower BBPowerSummarizer   --splits_list=/mnt/extraspace/susanna/BBMoments/Simulations_Moments_varStd/sim_ns256_seed${seed}_stdd${std}_stds${std}_gdm3.0_gsm3.0_msk_B/splits_list.txt   --bandpasses_list=./test_BBMoments_simulations/bandpasses.txt   --cells_fiducial=/mnt/extraspace/susanna/BBMoments/Simulations_Moments_varStd/sim_ns256_seed${seed}_stdd${std}_stds${std}_gdm3.0_gsm3.0_msk_B/cells_model.sacc   --cells_all_splits=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_all_splits.sacc   --cells_all_sims=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_all_sims.txt   --cells_coadded_total=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_coadded_total.sacc   --cells_coadded=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_coadded.sacc   --cells_noise=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_noise.sacc   --cells_null=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_null.sacc   --config=./test_BBMoments_simulations/config.yaml 
    done
done

# --------------------------------------
# BBCompsep with 500 sims
# both sync and dust varying
# --------------------------------------
for seed in {1000..1010}
do
    for std in 30
    do
	echo "BBPCompSep for seed = "${seed}", std = "${std}
	addqueue -s -q berg -m 2 -n 1x12 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_coadded.sacc   --cells_noise=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_noise.sacc   --cells_fiducial=/mnt/extraspace/susanna/BBMoments/Simulations_Moments_varStd/sim_ns256_seed${seed}_stdd${std}_stds${std}_gdm3.0_gsm3.0_msk_B/cells_model.sacc   --param_chains=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/param_chains.npz   --config_copy=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/config_copy.yml   --config=./test_BBMoments_simulations/config.yaml 
    done
done

# --------------------------------------
# BBPlotter with 500 sims
# both sync and dust varying
# --------------------------------------
for seed in {1000..1010}
do
    for std in 30
    do
	echo "BBPCompSep for seed = "${seed}", std = "${std}
	addqueue -q berg -m 10 /usr/bin/python3 -m bbpower BBPlotter   --cells_coadded_total=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_coadded_total.sacc   --cells_coadded=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_coadded.sacc   --cells_noise=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_noise.sacc   --cells_null=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/cells_null.sacc   --cells_fiducial=/mnt/extraspace/susanna/BBMoments/Simulations_Moments_varStd/sim_ns256_seed${seed}_stdd${std}_stds${std}_gdm3.0_gsm3.0_msk_B/cells_model.sacc   --param_chains=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/param_chains.npz   --plots=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/plots.dir   --plots_page=/mnt/extraspace/susanna/BBMoments_outputs_10varsims_COMPLETO/Output_BBMoments_seed${seed}_stdD${std}_stdS${std}/plots_page.html   --config=./test_BBMoments_simulations/config.yaml
    done
done
