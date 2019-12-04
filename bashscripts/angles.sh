#~/usr/bin/bash

fdir="./updated_runs/bias_angles/"
for ((j=1; j<=6; j++))
do 
    # per channel
    sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[-1.\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_angle"$j"_-1.0.npz"

    sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[1.\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_angle"$j"_1.0.npz"

    cp $fdir"config.yml" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_angle"$j"_0.0.npz"

    for ((k=1; k<=9; k++))
    do
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[-0.$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_angle"$j"_-0.$k.npz"

        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[0.$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_angle"$j"_0.$k.npz"
    done
done

for j in 1 3 5
do 
    jj=$((j+1))

    # symmetric
    sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[1.\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[1.\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/symmetric_eb_angle"$j$jj"_1.0.npz"

    sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[-1.\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[-1.\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/symmetric_eb_angle"$j$jj"_-1.0.npz"

    cp $fdir"config.yml" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/symmetric_eb_angle"$j$jj"_0.0.npz"

    for ((k=1; k<=9; k++))
    do
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[-0.$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[-0.$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/symmetric_eb_angle"$j$jj"_-0.$k.npz"

        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[0.$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[0.$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/symmetric_eb_angle"$j$jj"_0.$k.npz"
    done

    # asymmetric
    sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[-1.\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[1.\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_angle"$j$jj"_-1.0_1.0.npz"

    sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[1.\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[-1.\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_angle"$j$jj"_1.0_-1.0.npz"

    cp $fdir"config.yml" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_angle"$j$jj"_0.0_0.0.npz"

    for ((k=1; k<=9; k++))
    do
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[-0.$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[0.$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_angle"$j$jj"_-0."$k"_0.$k.npz"

        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[0.$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[-0.$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_angle"$j$jj"_0."$k"_-0.$k.npz"
    done
done
