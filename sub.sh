#!/bin/bash
while [ $# -gt 0 ]; do
    if [[ $1 == "-"* ]]; then
        v="${1/-/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

# -p : name cluster
# -g : nb gpu
# -w : walltime
# -b : batchsize
# -r : checkpoint file or dir
# -y : config file
# -q : name queue
# -t : time start ('2023-04-24 19:00:00')

if [ -z "$w" ];
then
    wtime=60
else
    wtime=$w
fi


if [ -z "$y" ];
then
    y=configs/autoencoder/vqgan.yaml
fi


if [ -z "$q" ];
then
    q="-q production"
else
    q="-t exotic -t night"
fi


if [ -z "$t" ];
then
    stime=''
else
    stime="-r '${t}'" 
fi



gpus=$(($g-1))
gseq=$(seq -s ',' 0 $gpus)

echo -p $p
echo -g $g
echo -gseq $gseq
echo -w $w
echo -b $b
echo -r "'$r'"
echo -y $y
echo -q $q
echo -t $stime
echo

# echo oarsub $q -p $p -l host=1/gpu=$g,walltime=$wtime --notify mail:cyril.regan@loria.fr $stime   "cd ~/GENS/FORK_stable-diffusion ; $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base $y -t --gpus $gseq --batch_size $b --resume '$r' ; sleep infinity"
# echo 
# exit N


if [ -z "$p" ];
then
    if [ -z "$r" ];
    then
        oarsub $q -l host=1/gpu=$g,walltime=$wtime --notify mail:cyril.regan@loria.fr  $stime   "cd ~/GENS/FORK_stable-diffusion ; $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base $y -t --gpus $gseq --batch_size $b ; sleep infinity"
    else
        oarsub $q -l host=1/gpu=$g,walltime=$wtime --notify mail:cyril.regan@loria.fr $stime "cd ~/GENS/FORK_stable-diffusion ; $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base $y -t --gpus $gseq --batch_size $b --resume '$r' ; sleep infinity"
    fi
else
    if [ -z "$r" ];
    then
        oarsub $q -p $p -l host=1/gpu=$g,walltime=$wtime --notify mail:cyril.regan@loria.fr $stime "cd ~/GENS/FORK_stable-diffusion ; $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base $y -t --gpus $gseq --batch_size $b  ; sleep infinity"
    else
        oarsub $q -p $p -l host=1/gpu=$g,walltime=$wtime --notify mail:cyril.regan@loria.fr $stime  "cd ~/GENS/FORK_stable-diffusion ; $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base $y -t --gpus $gseq --batch_size $b --resume '$r'  ; sleep infinity"
    fi
fi
