### run finetuning
CUDA_VISIBLE_DEVICES=0 python ./defense/ft/ft.py --result_file cifar10/badnet_resnet18 --yaml_path ./config/defense/ft/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --ratio 1.0 --lr 0.01 --log /log/cifar10/ft/

### run finepruning
CUDA_VISIBLE_DEVICES=0 python ./defense/fp/fp.py --result_file cifar10/badnet_resnet18 --yaml_path ./config/defense/fp/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --ratio 1.0 --log /log/cifar10/fp/

### run i-bau
CUDA_VISIBLE_DEVICES=0 python ./defense/i-bau/i-bau.py --result_file cifar10/badnet_resnet18 --yaml_path ./config/defense/i-bau/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --ratio 1.0 --lr 0.0001 --log /log/cifar10/i-bau/

### run mcr
CUDA_VISIBLE_DEVICES=0 python ./defense/mcr/mcr.py --result_file cifar10/badnet_resnet18 --yaml_path ./config/defense/mcr/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --ratio 1.0 --lr 0.01 --log /log/cifar10/mcr/

### run NAD
CUDA_VISIBLE_DEVICES=0 python ./defense/nad/nad.py --result_file cifar10/badnet_resnet18 --yaml_path ./config/defense/nad/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --ratio 1.0 --lr 0.01 --log /log/cifar10/nad/

### run ANP
CUDA_VISIBLE_DEVICES=0 python ./defense/anp/anp.py --result_file cifar10/badnet_resnet18 --yaml_path ./config/defense/anp/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --pruning_by threshold --ratio 1.0 --log /log/cifar10/anp/

### run our method
CUDA_VISIBLE_DEVICES=0 python ./defense/ft/distillation_dropout_increase.py --result_file cifar10/badnet_resnet18 --yaml_path ./config/defense/ft/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --ratio 1.0 --lr 0.01 --log /log/cifar10/dp_increase/ --layerwise_ratio 0.01 0.01 0.03 0.09 0.27 0.10

