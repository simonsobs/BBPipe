#~/usr/bin/bash

fdir="./updated_runs/bias_shifts/pairwise/"
for j in 1 3 5
do 
    jj=$((j+1))

    sed "s/shift\_$j: \['shift', 'fixed', \[0.\]\]/shift\_$j: \['shift', 'fixed', \[0.1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/shift\_$jj: \['shift', 'fixed', \[0.\]\]/shift\_$jj: \['shift', 'fixed', \[0.1\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_shift"$j$jj"_0.10.npz"

    sed "s/shift\_$j: \['shift', 'fixed', \[0.\]\]/shift\_$j: \['shift', 'fixed', \[-0.1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/shift\_$jj: \['shift', 'fixed', \[0.\]\]/shift\_$jj: \['shift', 'fixed', \[-0.1\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_shift"$j$jj"_-0.10.npz"

    for ((k=0; k<=9; k++))
    do
        sed "s/shift\_$j: \['shift', 'fixed', \[0.\]\]/shift\_$j: \['shift', 'fixed', \[-0.0$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/shift\_$jj: \['shift', 'fixed', \[0.\]\]/shift\_$jj: \['shift', 'fixed', \[-0.0$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_shift"$j$jj"_-0.0$k.npz"
    done
    for ((k=0; k<=9; k++))
    do
        sed "s/shift\_$j: \['shift', 'fixed', \[0.\]\]/shift\_$j: \['shift', 'fixed', \[0.0$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/shift\_$jj: \['shift', 'fixed', \[0.\]\]/shift\_$jj: \['shift', 'fixed', \[0.0$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_shift"$j$jj"_0.0$k.npz"
    done
done


