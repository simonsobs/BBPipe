#!/bin/bash

nside=512
sens=1
knee=1
nsplits=4
for i in {1..10}
do
    echo $i
    python3 generate_SO_maps.py ${i} ${nside} ${sens} ${knee} ${nsplits}
done
