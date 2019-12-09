#~/usr/bin/bash

fdir="./updated_runs/bias_angles/"

xmin=-0.02
xmax=0.02
dx=0.002

# per channel
for ((j=1; j<=6; j++))
do 
    for k in {0..20}
    do
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_angle"$j"_$x1.npz"
    done
done

for j in 1 3 5
do 
    jj=$((j+1))

    for k in {0..20}
    do
        # symmetric
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[$x1\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/symmetric_eb_angle"$j$jj"_$x1.npz"

        # asymmetric
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        x2=$(echo "scale=3; $xmax-$k*$dx" | bc)
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[$x2\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_angle"$j$jj"_"$x1"_"$x2".npz"
    done
done

