#~/usr/bin/bash

fdir="./updated_runs/bias_gains/perchannel/"
for ((j=1; j<=6; j++))
do 
    cp $fdir"config.yml" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_gain"$j"_1.00.npz"

    for ((k=0; k<=5; k++))
    do
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[1.1$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_gain"$j"_1.1$k.npz"
    done

    for ((k=5; k<=9; k++))
    do
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[0.8$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_gain"$j"_0.8$k.npz"
    done

    for ((k=0; k<=9; k++))
    do
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[0.9$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_gain"$j"_0.9$k.npz"

        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[1.0$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_gain"$j"_1.0$k.npz"
    done
done
