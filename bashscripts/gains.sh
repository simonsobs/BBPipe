#~/usr/bin/bash

fdir="./updated_runs/bias_gains/"

xmin=0.98
xmax=1.02
dx=0.002

#nk=$(echo "scale=0; ($xmax-$xmin)/$dx" | bc)
#echo $nk

# per channel
for ((j=1; j<=6; j++))
do 
    for k in {0..20}
    do
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/perchannel_eb_gain"$j"_$x1.npz"
    done
done

for j in 1 3 5
do 
    jj=$((j+1))

    # symmetric
    for k in {0..20}
    do
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[$x1\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/symmetric_eb_gain"$j$jj"_$x1.npz"
    done

    # asymmetric
    for k in {0..20}
    do
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        x2=$(echo "scale=3; $xmax-$k*$dx" | bc)
        sed "s/gain\_$j: \['gain', 'fixed', \[1.\]\]/gain\_$j: \['gain', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/gain\_$jj: \['gain', 'fixed', \[1.\]\]/gain\_$jj: \['gain', 'fixed', \[$x2\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/sampler_out.npz" $fdir"autoresults/asymmetric_eb_gain"$j$jj"_"$x1"_"$x2".npz"
    done
done
