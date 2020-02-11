#~/usr/bin/bash

fdir="/mnt/zfsusers/mabitbol/BBPipe/updated_runs/multiruntest/"
exdir="/mnt/zfsusers/mabitbol/BBPipe/examples/data/"


# gains
xmin=0.9
xmax=1.10
dx=0.001
nr=200

for ((j=1; j<=6; j++))
do 
    for k in {0..$nr}
    do
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        configfile=$fdir"configs/config_perchannel_gain"$j"_$x1.yml"
        paramsfile=$fdir"autoresults/perchannel_gain"$j"_$x1.npz"
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml
    done
done

for j in 1 3 5
do 
    jj=$((j+1))
    for k in {0..$nr}
    do
        # symmetric
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        configfile=$fdir"configs/config_symmetric_gain"$j$jj"_$x1.yml"
        paramsfile=$fdir"autoresults/symmetric_gain"$j$jj"_$x1.npz"
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[$x1\]\]/" $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml

        # asymmetric
        x2=$(echo "scale=3; $xmax-$k*$dx" | bc)
        configfile=$fdir"configs/config_asymmetric_gain"$j$jj"_"$x1"_"$x2".yml"
        paramsfile=$fdir"autoresults/asymmetric_gain"$j$jj"_"$x1"_"$x2".npz"
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[$x2\]\]/" $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml
    done
done

# shifts
xmin=-0.02
xmax=0.02
dx=0.001
nr=200

# per channel
for ((j=1; j<=6; j++))
do 
    for k in {0..$nr}
    do
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        configfile=$fdir"configs/config_perchannel_shift"$j"_$x1.yml"
        paramsfile=$fdir"autoresults/perchannel_shift"$j"_$x1.npz"
        sed "s/shift\_$j: \['shift', 'fixed', \[0.\]\]/shift\_$j: \['shift', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml
    done
done

for j in 1 3 5
do 
    jj=$((j+1))

    for k in {0..$nr}
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

# angles
xmin=-0.5
xmax=0.5
dx=0.05

# per channel
for ((j=1; j<=6; j++))
do 
    for k in {0..$nr}
    do
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        configfile=$fdir"configs/config_perchannel_angle"$j"_$x1.yml"
        paramsfile=$fdir"autoresults/perchannel_angle"$j"_$x1.npz"
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml
    done
done

for j in 1 3 5
do 
    jj=$((j+1))
    for k in {0..$nr}
    do
        # symmetric
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        configfile=$fdir"configs/config_symmetric_angle"$j$jj"_$x1.yml"
        paramsfile=$fdir"autoresults/symmetric_angle"$j$jj"_$x1.npz"
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[$x1\]\]/" $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml

        # asymmetric
        x2=$(echo "scale=3; $xmax-$k*$dx" | bc)
        configfile=$fdir"configs/config_asymmetric_angle"$j$jj"_"$x1"_"$x2".yml"
        paramsfile=$fdir"autoresults/asymmetric_angle"$j$jj"_"$x1"_"$x2".npz"
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[$x2\]\]/" $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml
    done
done

# dphi1
xmin=-80
xmax=80
dx=8

# per channel
for ((j=1; j<=6; j++))
do 
    for k in {0..$nr}
    do
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        configfile=$fdir"configs/config_perchannel_dphi1"$j"_$x1.yml"
        paramsfile=$fdir"autoresults/perchannel_dphi1"$j"_$x1.npz"
        sed "s/dphi1\_$j: \['dphi1', 'fixed', \[0.\]\]/dphi1\_$j: \['dphi1', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml
    done
done

for j in 1 3 5
do 
    jj=$((j+1))

    for k in {0..$nr}
    do
        # symmetric
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        configfile=$fdir"configs/config_symmetric_dphi1"$j$jj"_$x1.yml"
        paramsfile=$fdir"autoresults/symmetric_dphi1"$j$jj"_$x1.npz"
        sed "s/dphi1\_$j: \['dphi1', 'fixed', \[0.\]\]/dphi1\_$j: \['dphi1', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        sed -i "s/dphi1\_$jj: \['dphi1', 'fixed', \[0.\]\]/dphi1\_$jj: \['dphi1', 'fixed', \[$x1\]\]/" $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml

        # asymmetric
        x2=$(echo "scale=3; $xmax-$k*$dx" | bc)
        configfile=$fdir"configs/config_asymmetric_dphi1"$j$jj"_"$x1"_"$x2".yml"
        paramsfile=$fdir"autoresults/asymmetric_dphi1"$j$jj"_"$x1"_"$x2".npz"
        sed "s/dphi1\_$j: \['dphi1', 'fixed', \[0.\]\]/dphi1\_$j: \['dphi1', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $configfile
        sed -i "s/dphi1\_$jj: \['dphi1', 'fixed', \[0.\]\]/dphi1\_$jj: \['dphi1', 'fixed', \[$x2\]\]/" $configfile
        addqueue -q cmb -c "1 day" -m 4 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0.sacc"   --cells_noise=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_noise.sacc"   --cells_fiducial=$exdir"SO_V3_Mock4_r0.01_phase0_angle0_sinuous0_eb0_fiducial.sacc"   --config=$configfile   --params_out=$paramsfile   --config_copy=updated_runs/multiruntest/output/config_copy.yml
    done
done

