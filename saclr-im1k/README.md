# SACLR

### Prepare imagenet dataset:
    # download ImageNet ILSVRC2012 from ImageNet-website
    # Prepare ImageNet folders, for example with the help of extract_ILSVRC.sh

### Prepare imagenet100 dataset:
    - python make_imagenet100.py full/imagenet/path desired/imagenet100/path 


## Usage SACLR
- See slurm-scripts
###
#### ImageNet1k
##### SACLR-All multi-crop
```
python main_saclr_multicrop.py --method saclr-all-cosine --rho 0.99999 --alpha 0.01 --temp 0.15 --s_init_t 0.5 --data_path /cluster/work/imagenet/ --epochs 200 --batch_size 256 --savedir mcropalle200b256_run --num_worker 10 --checkpoint_interval 5

```

##### SACLR-Mix multi-crop
```
python main_saclr_multicrop.py --method saclr-all-batchmix-cosine --rho 0.99999 --alpha 0.01 --temp 0.15 --s_init_t 0.5 --data_path /cluster/work/imagenet/ --epochs 800 --batch_size 1024 --savedir mcropbmixe800 --num_worker 10 --checkpoint_interval 1

```



## Usage evaluation

#### Linear classifier evaluation ImageNet
```
python evaluate_linear.py --data_path ~/Datasets/imagenet/ --backbone_path logs/resnet50_last.pth --savedir linear
```

#### finetune evaluation ImageNet
```
python evaluate_finetune.py --data_path ~/Datasets/imagenet/ --backbone_path saclr-im1k/sbmix05lr05alpha0001rho099-ep1000/resnet50_last.pth --train-percent 1 --savedir finetune-1-sbmix05 --lr_backbone 0.0005 --lr_linear 0.05 --epochs 30
```
#### kNN evaluation ImageNet
```
python evaluate_nn.py --data_path ~/Datasets/imagenet/ --backbone_path logs/resnet50_last.pth
```
