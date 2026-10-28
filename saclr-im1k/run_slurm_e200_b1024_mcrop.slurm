#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=<>
#SBATCH --time=7-00:30:00

#SBATCH --nodes=1

#SBATCH --gres=gpu:4  
#SBATCH --mem=150G      
#SBATCH --gpus-per-node=4      
#SBATCH --cpus-per-task=10      
#SBATCH --tasks-per-node=4      

#SBATCH --job-name="mcrop"

#SBATCH --constraint="h100|a100"
#SBATCH --constraint=gpu80g


module purge
module load Python/3.11.5-GCCcore-13.2.0

source venv/bin/activate
# pip install -r requirements.txt
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

python main_saclr_multicrop.py --method saclr-all-batchmix-cosine --rho 0.99999 --alpha 0.01 --temp 0.15 --s_init_t 0.5 --data_path /cluster/work/imagenet/ --epochs 200 --batch_size 1024 --savedir mcropbmixe200b1024 --num_worker 10 --checkpoint_interval 5
