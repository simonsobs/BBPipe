#!/bin/bash

nside=64
sens=1
knee=1
nsplits=4
mask_type=analytic
for i in {1..1}
do
    echo $i
    python3 generate_SO_maps.py ${i} ${nside} ${sens} ${knee} ${nsplits} ${mask_type}
    printf -v mockno "%04d" $i
    echo "./examples/SO_V3_ns${nside}_sens${sens}_knee${knee}_${nsplits}split_Mask${mask_type}_Mock${mockno}" >> list_sims.txt
done
