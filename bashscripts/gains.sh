#~/usr/bin/bash

fdir="./updated_runs/bias_gains/"

for ((j=1; j<=6; j++))
do 
    # per channel
    cp $fdir"config.yml" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_gain"$j"_1.00.npz"

    for ((k=0; k<=5; k++))
    do
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[1.1$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_gain"$j"_1.1$k.npz"
    done

    for ((k=5; k<=9; k++))
    do
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[0.8$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_gain"$j"_0.8$k.npz"
    done

    for ((k=0; k<=9; k++))
    do
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[0.9$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_gain"$j"_0.9$k.npz"

        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[1.0$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_gain"$j"_1.0$k.npz"
    done
done

for j in 1 3 5
do 
    jj=$((j+1))

    # symmetric
    sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[1.1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[1.1\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/symmetric_eb_gain"$j$jj"_1.10.npz"

    for ((k=0; k<=9; k++))
    do
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[0.9$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[0.9$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/symmetric_eb_gain"$j$jj"_0.9$k.npz"

        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[1.0$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[1.0$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/symmetric_eb_gain"$j$jj"_1.0$k.npz"
    done

    # asymmetric
    sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[0.9\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[1.1\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_gain"$j$jj"_0.90_1.10.npz"

    sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[1.1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[0.9\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_gain"$j$jj"_1.10_0.90.npz"

    cp $fdir"config.yml" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_gain"$j$jj"_1.00_1.00.npz"

    for ((k=1; k<=9; k++))
    do
        i=$((10-k)) 
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[0.9$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[1.0$i\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_gain"$j$jj"_0.9"$k"_1.0$i.npz"

        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[1.0$i\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[0.9$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_gain"$j$jj"_1.0"$i"_0.9$k.npz"
    done

done
