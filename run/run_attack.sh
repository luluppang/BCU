### run badnet
CUDA_VISIBLE_DEVICES=0 python ./attack/badnet_attack.py --yaml_path ../config/attack/badnet/cifar10.yaml --dataset cifar10 --dataset_path ../../../../data --save_folder_name cifar10/badnet_resnet18 --random_seed 0 --model resnet18 --attack_target 0 --epochs 200 --lr_scheduler MultiStepLR --steplr_milestones 100 150 --steplr_gamma 0.1 --lr 0.01 --pratio 0.1 --patch_mask_path ../resource/badnet/cifar10_bottom_right_3by3_blackwhite.npy

### run blended
CUDA_VISIBLE_DEVICES=0 python ./attack/blended_attack.py --yaml_path ../config/attack/blended/cifar10.yaml --dataset cifar10 --dataset_path ../../../../data --save_folder_name cifar10/blended_resnet18 --random_seed 0 --model resnet18 --attack_target 0 --epochs 200 --lr_scheduler MultiStepLR --steplr_milestones 100 150 --steplr_gamma 0.1 --lr 0.01 --pratio 0.1

### run lc
CUDA_VISIBLE_DEVICES=0 python ./attack/lc_attack.py --yaml_path ../config/attack/lc/cifar10.yaml --dataset cifar10 --dataset_path ../../../../data --save_folder_name cifar10/lc_resnet18 --random_seed 0 --model resnet18 --attack_target 0 --epochs 200 --lr_scheduler MultiStepLR --steplr_milestones 100 150 --steplr_gamma 0.1 --lr 0.01

### run sig
CUDA_VISIBLE_DEVICES=0 python ./attack/sig_attack.py --yaml_path ../config/attack/sig/cifar10.yaml --dataset cifar10 --dataset_path ../../../../data --save_folder_name cifar10/sig_resnet18 --random_seed 0 --model resnet18 --attack_target 0 --epochs 200 --lr_scheduler MultiStepLR --steplr_milestones 100 150 --steplr_gamma 0.1 --lr 0.01

### run IAB
CUDA_VISIBLE_DEVICES=0 python ./attack/inputaware_attack.py --yaml_path ../config/attack/inputaware/cifar10.yaml --dataset cifar10 --dataset_path ../../../../data --save_folder_name cifar10/IAB_resnet18 --random_seed 0 --model resnet18 --target_label 0 --epochs 150

### run wanet
CUDA_VISIBLE_DEVICES=0 python ./attack/wanet_attack.py --yaml_path ../config/attack/wanet/cifar10.yaml --dataset cifar10 --dataset_path ../../../../data --save_folder_name cifar10/wanet_resnet18 --random_seed 0 --model resnet18 --target_label 0
