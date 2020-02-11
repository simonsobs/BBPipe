#~/usr/bin/bash

fdir="/mnt/zfsusers/mabitbol/BBPipe/updated_runs/multiruntest/"
exdir="/mnt/zfsusers/mabitbol/BBPipe/examples/data/"

xmin=-0.05
xmax=0.05
dx=0.005

# per channel
for ((j=1; j<=6; j++))
do 
    for k in {0..20}
    do
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        configfile=$fdir"configs/config_perchannel_shift_"$j"_$x1.yml"
        paramsfile=$fdir"autoresults/perchannel_shift"$j"_$x1.npz"
        sed "s/shift\_$j: \['shift', 'fixed', \[0.\]\]/shift\_$j: \['shift', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml
    done
done

for j in 1 3 5
do 
    jj=$((j+1))

    for k in {0..20}
    do
        # symmetric
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        configfile=$fdir"configs/config_symmetric_shift"$j$jj"_$x1.yml"
        paramsfile=$fdir"autoresults/symmetric_shift"$j$jj"_$x1.npz"
        sed "s/shift\_$j: \['shift', 'fixed', \[0.\]\]/shift\_$j: \['shift', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        sed -i "s/shift\_$jj: \['shift', 'fixed', \[0.\]\]/shift\_$jj: \['shift', 'fixed', \[$x1\]\]/" $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml

        # asymmetric
        x2=$(echo "scale=3; $xmax-$k*$dx" | bc)
        configfile=$fdir"configs/config_asymmetric_shift"$j$jj"_"$x1"_"$x2".yml"
        paramsfile=$fdir"autoresults/asymmetric_shift"$j$jj"_"$x1"_"$x2".npz"
        sed "s/shift\_$j: \['shift', 'fixed', \[0.\]\]/shift\_$j: \['shift', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        sed -i "s/shift\_$jj: \['shift', 'fixed', \[0.\]\]/shift\_$jj: \['shift', 'fixed', \[$x2\]\]/" $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml
    done
done

