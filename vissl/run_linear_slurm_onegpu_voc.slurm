#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=<>
#SBATCH --time=2-00:30:00

#SBATCH --nodes=1

#SBATCH --gres=gpu:2  
#SBATCH --mem=128G      
#SBATCH --gpus-per-node=2      
#SBATCH --cpus-per-task=10      
#SBATCH --tasks-per-node=2      

#SBATCH --job-name="vissl_voc"
#SBATCH --constraint="h100|a100"


module purge
#module load Python/3.11.5-GCCcore-13.2.0

module load Anaconda3/2023.09-0
conda activate vissl_env_cuda111 #vissl_env 
# source venv/bin/activate
# pip install -r requirements.txt
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121


# example
python tools/train_svm.py \
  config=benchmark/linear_image_classification/voc07/eval_resnet_8gpu_transfer_voc07_svm \
  config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/resnet50_last.pth"\
  config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME="" \
  config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk.base_model._feature_blocks."\
  config.DATA.TRAIN.DATA_PATHS="[\"vissl/voc2007/train_images.npy\"]" \
  config.DATA.TRAIN.LABEL_PATHS="[\"vissl/voc2007/train_labels.npy\"]" \
  config.DATA.TEST.DATA_PATHS="[\"vissl/voc2007/test_images.npy\"]" \
  config.DATA.TEST.LABEL_PATHS="[\"vissl/voc2007/test_labels.npy\"]" \
  config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=128 \
  config.DATA.TEST.BATCHSIZE_PER_REPLICA=128 \
  config.DISTRIBUTED.NUM_NODES=1 \
  config.DISTRIBUTED.NUM_PROC_PER_NODE=2 \
  config.CHECKPOINT.DIR="checkpoint_voc_v5"\
  config.PROFILING.OUTPUT_FOLDER="checkpoint_voc_v5"
