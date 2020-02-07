#~/usr/bin/bash

fdir="./updated_runs/bias_shifts/"

xmin=-0.05
xmax=0.05
dx=0.005

# per channel
for ((j=1; j<=6; j++))
do 
    for k in {0..20}
    do
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        sed "s/shift\_$j: \['shift', 'fixed', \[0.\]\]/shift\_$j: \['shift', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/params_out.npz" $fdir"autoresultsr/perchannel_eb_shift"$j"_$x1.npz"
    done
done

for j in 1 3 5
do 
    jj=$((j+1))

    for k in {0..20}
    do
        # symmetric
        x1=$(echo "scale=3; $xmin+$k*$dx" | bc)
        sed "s/shift\_$j: \['shift', 'fixed', \[0.\]\]/shift\_$j: \['shift', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/shift\_$jj: \['shift', 'fixed', \[0.\]\]/shift\_$jj: \['shift', 'fixed', \[$x1\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/params_out.npz" $fdir"autoresultsr/symmetric_eb_shift"$j$jj"_$x1.npz"

        # asymmetric
        x2=$(echo "scale=3; $xmax-$k*$dx" | bc)
        sed "s/shift\_$j: \['shift', 'fixed', \[0.\]\]/shift\_$j: \['shift', 'fixed', \[$x1\]\]/" $fdir"config.yml" > $fdir"runconfig.yml"
        sed -i "s/shift\_$jj: \['shift', 'fixed', \[0.\]\]/shift\_$jj: \['shift', 'fixed', \[$x2\]\]/" $fdir"runconfig.yml"
        /mnt/zfsusers/mabitbol/.local/lib/python3.6/site-packages/bbpipe $fdir"settings.yml"
        cp $fdir"output/params_out.npz" $fdir"autoresultsr/asymmetric_eb_shift"$j$jj"_"$x1"_"$x2".npz"
    done
done

