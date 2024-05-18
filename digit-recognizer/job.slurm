#!/bin/bash
#SBATCH -n 32
#SBATCH --exclude=hepnode2
#SBATCH --output=log/decisiontree/log.txt

echo "==="
cat job.slurm
echo "==="

python3 main.py dt

echo "finished"