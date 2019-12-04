#~/usr/bin/bash

fdir="./updated_runs/bias_angles/perchannel/"
for ((j=1; j<=6; j++))
do 
    sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[-1.\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_angle"$j"_-1.0.npz"

    sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[1.\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_angle"$j"_1.0.npz"

    for ((k=1; k<=9; k++))
    do
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[-0.$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_angle"$j"_-0.$k.npz"
    done
    for ((k=1; k<=9; k++))
    do
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[0.$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_angle"$j"_0.$k.npz"
    done
done
