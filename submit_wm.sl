#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -x bart14,bart13,eos20,eos21
#SBATCH --time=24:00:00
#SBATCH -o /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out  # Write the log on tmp1
#SBATCH -e /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out

python data/generation_script.py --rollouts 1000 --rootdir datasets/carnav --threads 8
python trainvae.py --logdir exp_dir
python trainmdrnn.py --logdir exp_dir
python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
python test_controller.py --logdir exp_dir
