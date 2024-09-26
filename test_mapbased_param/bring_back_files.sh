tags = 'knee1_newmask_apo10_nlb5_sens2' 'knee1_newmask_apo10_nlb5_r0p01' 'wn_newmask_apo10_nlb5_0' 'wn_newmask_apo10_nlb5_AL0p5' 'WN_newmask_apo10_nlb5_r0p01' 'wn_newmask_apo10_nlb5_no_dust_marg' 'knee1_newmask_apo10_nlb5' 'knee0_newmask_apo10_nlb5' 
#'WN_external_gaussian_sims'
for VARIABLE in $tags
do
	for i in 0 1 2 3 4 5 6 7 8 9
	do
	mkdir $VARIABLE
	scp -r josquin@cori.nersc.gov:/global/cscratch1/sd/josquin/SO_pipe/results_F2F_Jun19/$VARIABLE/*txt $VARIABLE/
	scp -r josquin@cori.nersc.gov:/global/cscratch1/sd/josquin/SO_pipe/results_F2F_Jun19/$VARIABLE/*pdf $VARIABLE/
	done	
done