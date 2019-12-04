#~/usr/bin/bash

fdir="./updated_runs/bias_angles/opppair/"
for j in 1 3 5
do 
    for ((k=1; k<=9; k++))
    do
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[-0.$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[0.$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_angle"$j$jj"_-0."$k"_0.$k.npz"
    done

    for ((k=1; k<=9; k++))
    do
        sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[0.$k\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[-0.$k\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_angle"$j$jj"_0."$k"_-0.$k.npz"
    done

done

for j in 1 3 5
do 
    jj=$((j+1))
    sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[-1.\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[1.\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_angle"$j$jj"_-1.0_1.0.npz"

    sed "s/angle\_$j: \['angle', 'fixed', \[0.\]\]/angle\_$j: \['angle', 'fixed', \[1.\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
    sed -i "s/angle\_$jj: \['angle', 'fixed', \[0.\]\]/angle\_$jj: \['angle', 'fixed', \[-1.\]\]/" $fdir"runconfig.yml"
    /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
    cp $fdir"output/sampler_out.npz" $fdir"autoresults/eb_angle"$j$jj"_1.0_-1.0.npz"
done
