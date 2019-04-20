#!/bin/bash

OPTION=$1

if [ "$OPTION" == "send" ]; then
    bsub -n 6 -W 4:00 -R "rusage[mem=1024, ngpus_excl_p=1]" python cil.py
elif [ "$OPTION" == "watch" ]; then
    watch -n 1 bjobs
elif [ "$OPTION" == "module" ]; then 
	module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy
else
    echo "Unknown option"
fi
