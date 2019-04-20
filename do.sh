#!/bin/bash

OPTION=$1

if [ "$OPTION" == "send" ]; then
    bsub -n 6 -W 4:00 -R "rusage[mem=1024, ngpus_excl_p=1]" python cil.py
elif [ "$OPTION" == "watch" ]; then
    watch -n 1 bjobs
else
    echo "Unknown option"
fi
