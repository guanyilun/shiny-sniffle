#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -J srcmask
#SBATCH -t 06:00:00
#SBATCH --qos short
#SBATCH --partition physics
# defined externally

module load enki_yilun

dataset=s19_galactic_new
mask=mask_mouse.fits  # srcs+dust


depot=/scratch/gpfs/yilung/gc_data/tmasks
tag=s19_f090_gal_srcmask_mouse
ntask=10
nomp=4
export DISABLE_MPI=true
for ((i=0;i<ntask;i++)); do
    OMP_NUM_THREADS=${nomp} mask2cuts.py "s19,f090,/all" ${mask} --gal --depot ${depot} --tag ${tag} --dataset ${dataset} --fmpi --frank ${i} --fsize ${ntask} --logfile s19_gal_mouse --buffer 0 --force -v 2 & sleep 1
done

wait
