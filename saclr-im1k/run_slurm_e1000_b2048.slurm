#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=<>
#SBATCH --time=14-00:30:00

#SBATCH --nodes=1

#SBATCH --gres=gpu:4  
#SBATCH --mem=192G      
#SBATCH --gpus-per-node=4      
#SBATCH --cpus-per-task=10      
#SBATCH --tasks-per-node=4      

#SBATCH --job-name="bmixep200"

#SBATCH --constraint="h100|a100"
#SBATCH --constraint=gpu80g


module purge
module load Python/3.11.5-GCCcore-13.2.0

source venv/bin/activate
# pip install -r requirements.txt
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121


python main_saclr_multigpu.py --method saclr-all-batchmix-cosine --rho 0.99999 --alpha 0.01 --temp 0.15 --s_init_t 0.5 --data_path /cluster/work/imagenet/ --epochs 1000 --batch_size 2048 --savedir bmixe1000b2048 --num_worker 10 --lr_scale linear --base_lr 0.5 --checkpoint_interval 5
python main_saclr_multigpu.py --method saclr-all-cosine --rho 0.99999 --alpha 0.01 --temp 0.15 --s_init_t 0.5 --data_path /cluster/work/imagenet/ --epochs 1000 --batch_size 2048 --savedir bmixe1000b2048 --num_worker 10 --lr_scale linear --base_lr 0.5 --checkpoint_interval 5
python main_saclr_multigpu.py --method saclr-1-cosine --rho 0.999 --alpha 0.05 --temp 0.2 --s_init_t -1.0 --data_path /cluster/work/imagenet/ --epochs 200 --batch_size 256 --savedir onee1000b2048 --num_workers 10 --lr_scale linear --base_lr 0.5 --checkpoint_interval 5
