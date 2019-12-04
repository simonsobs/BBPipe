#~/usr/bin/bash

fdir="./updated_runs/bias_gains/pairwise/"
for j in 1 3 5
do 
    jj=$((j+1))
    sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[1.1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[1.1\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_gain"$j$jj"_1.10.npz"

    for ((k=0; k<=9; k++))
    do
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[0.9$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[0.9$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_gain"$j$jj"_0.9$k.npz"
    done

    for ((k=1; k<=9; k++))
    do
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[1.0$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[1.0$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_gain"$j$jj"_1.0$k.npz"
    done
done

