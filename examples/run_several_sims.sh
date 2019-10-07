#!/bin/bash

#python_exec="python3"
python_exec="addqueue -q cmb -s -m 2 -n 1x8 /usr/bin/python3"
nside=512
sens=1
knee=1
nsplits=4
mask_type=nhits
for i in {2..500}
do
    echo $i
    ${python_exec} generate_SO_maps.py ${i} ${nside} ${sens} ${knee} ${nsplits} ${mask_type}
    printf -v mockno "%04d" $i
    echo "/mnt/extraspace/damonge/SO/BBPipe/SO_V3_ns${nside}_sens${sens}_knee${knee}_${nsplits}split_Mask${mask_type}_Mock${mockno}" >> list_sims.txt
done
