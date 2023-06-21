#!/bin/sh

set -eu

CWD=stable_style_2
SCRIPTDIR=/home/cregan/Documents/CODE/Projets/4_GENS/FORK_stable-diffusion

docker run --rm --gpus=all \
-v $SCRIPTDIR:$SCRIPTDIR \
-it "$CWD" \
/bin/bash
echo "run ?" 
