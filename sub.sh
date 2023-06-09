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
    if [ "$q" = "exotic" ]; 
    then 
        q="-t exotic -t night"
        wtime=13
    else
        q=$q
    fi
fi

if [ -z "$p" ];
then
    pclus=''
else
    pclus="-p $p"
fi


if [ -z "$t" ];
then
    stime=''
else
    stime="-r '${t}'" 
fi



if [ -z "$r" ];
then 
    resume=''
else
    # cur_time=`date +"%Y-%m-%d_%H-%M"`
    # dir_name="ckpt_${cur_time}"
    # mkdir logs/$dir_name
    # mkdir logs/$dir_name/checkpoints
    # cp $r logs/$dir_name/checkpoints
    # checkpoint="logs/$dir_name/checkpoints/last.ckpt"
    resume="--resume ${checkpoint}" 
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

# echo oarsub $q $pclus -l host=1/gpu=$g,walltime=$wtime --notify mail:cyril.regan@loria.fr $stime "cd ~/GENS/FORK_stable-diffusion ; $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base $y -t --gpus $gseq --batch_size $b $resume ; sleep infinity"
# echo 
# exit N

# ". /etc/profile && module load singularity && mpirun -hostfile \$OAR_NODE_FILE --mca orte_rsh_agent oarsh -- `which singularity` exec my_mpi_image.sif /opt/mpitest"

oarsub $q $pclus -l host=1/gpu=$g,walltime=$wtime --notify mail:cyril.regan@loria.fr $stime ". /etc/profile ; module load singularity ; cd ~/GENS/FORK_stable-diffusion ;   $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base $y -t --gpus $gseq --batch_size $b $resume ; sleep infinity"