'''
This file is modified based on the following source:
link : https://github.com/kangliucn/Fine-pruning-defense
The defense method is called fp.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. add some addtional backbone such as resnet18 and vgg19
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. fp defense:
        a. hook the activation layer representation of each data
        b. rank the mean of activation for each neural
        c. according to the sorting results, prune and test the accuracy
        d. find the last model with reasonable ACC 
        e. finetune the model with validation data
    4. test the result and get ASR, ACC, RC 
'''


import argparse
import logging
import os
import sys 
sys.path.append('../../')
sys.path.append(os.getcwd())
#os.chdir(sys.path[0])
#sys.path.append('../../')
#os.getcwd()
print(os.getcwd())
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np

from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random 
from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils.save_load_attack import load_attack_result
import yaml
from pprint import pprint, pformat

def get_args():
    #set the basic parameter
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str)
    parser.add_argument('--checkpoint_save', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr') 

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--seed', type=str, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')
    parser.add_argument('--yaml_path', type=str, default="./config/defense/fp/config.yaml", help='the path of yaml')

    #set the parameter for the fp defense
    parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
    parser.add_argument('--acc_ratio', type=float, help='the tolerance ration of the clean accuracy')

    arg = parser.parse_args()

    print(arg)
    return arg


def test_epoch(arg, testloader, model, criterion, epoch, word):
    '''test the student model with regard to test data for each epoch
    arg:
        Contains default parameters
    testloader:
        the dataloader of clean test data or backdoor test data
    model:
        the training model
    criterion:
        criterion during the train process
    epoch:
        current epoch
    word:
        'bd' or 'clean'
    '''
    model.eval()

    total_clean, total_clean_correct, test_loss = 0, 0, 0

    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)

    # if word == 'bd':
    #     logging.info(f'test_Epoch{epoch}: asr:{avg_acc_clean}({total_clean_correct}/{total_clean})')
    #     #progress_bar(i, len(testloader), 'Test %s ASR: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
    # if word == 'clean':
    #     logging.info(f'test_Epoch{epoch}: clean_acc:{avg_acc_clean}({total_clean_correct}/{total_clean})')
    #     #progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))

    return test_loss / (i + 1), avg_acc_clean


class MaskedLayer(nn.Module):
    def __init__(self, in_channels, out_channels, mask):
        super(MaskedLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.masked_fc = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.mask = mask

    def forward(self, input):
        out = input * self.mask
        out = self.masked_fc(out)
        return out


def fp(args, result , config):
    ### set logger
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    if args.log_file_name is not None:
        fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + args.log_file_name + '.log')
    else:
        if args.log is not None and args.log != '':
            fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        else:
            fileHandler = logging.FileHandler(os.getcwd() + './log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    fix_random(args.seed)

    ### a. hook the activation layer representation of each data
    # Prepare model
    netC = generate_cls_model(model_name=args.model, num_classes=args.num_classes)
    netC.load_state_dict(result['model'])
    netC.to(args.device)
    netC.eval()
    netC.requires_grad_(False)

    criterion = nn.CrossEntropyLoss()
    # Prepare dataloader and check initial acc_clean and acc_bd
    tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train = True)
    x = result['clean_train']['x']
    y = result['clean_train']['y']
    data_all_length = len(y)
    ran_idx = choose_index(args, data_all_length) 
    log_index = os.getcwd() + args.log + 'index_' + args.log_file_name + '.txt'
    np.savetxt(log_index, ran_idx, fmt='%d')
    data_set = list(zip([x[ii] for ii in ran_idx], [y[ii] for ii in ran_idx]))
    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    trainloader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=True)
    x = result['clean_val']['x']
    y = result['clean_val']['y']
    data_all_length = len(y)
    ran_idx = choose_index(args, data_all_length)
    log_index = os.getcwd() + args.log + "index_val_" + str(args.ratio) + args.log_file_name + ".txt"
    np.savetxt(log_index, ran_idx, fmt='%d')
    data_set_val = list(zip([x[ii] for ii in ran_idx], [y[ii] for ii in ran_idx]))
    # data_set_val = list(zip(x, y))
    data_set_valset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set_val,
        poison_idx=np.zeros(len(data_set_val)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    val_loader = torch.utils.data.DataLoader(data_set_valset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True)

    tran = get_transform(args.dataset, *([args.input_height, args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x, y))
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=True, pin_memory=True)

    tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x, y))
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    testloader_bd = data_bd_loader
    testloader_clean = data_clean_loader
    for name, module in netC._modules.items():
        print(name)

    # Forward hook for getting layer's output
    global result_mid
    container = []
    result_mid = torch.tensor(0).to(args.device)
    with torch.no_grad():
        def forward_hook(module, input, output):
            global result_mid
            result_mid = output
            # print(result_mid.shape)
            container.append(output.detach().clone().cpu())

    if args.model == 'preactresnet18':
        hook = netC.layer4.register_forward_hook(forward_hook)
    if args.model == 'vgg16':
        hook = netC.features.register_forward_hook(forward_hook)
    if args.model == 'vgg19':
        hook = netC.features.register_forward_hook(forward_hook)
    if args.model == 'resnet18':
        hook = netC.layer4.register_forward_hook(forward_hook)
    if args.model == 'densenet161':
        hook = netC.features.register_forward_hook(forward_hook)
    if args.model == 'mobilenet_v3_large':
        hook = netC.features.register_forward_hook(forward_hook)
    if args.model == 'efficientnet_b3':
        hook = netC.features.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    logging.info("Forwarding all the training dataset:")
    with torch.no_grad():
        flag = 0
        for batch_idx, (inputs, _) in enumerate(val_loader):
            inputs = inputs.to(args.device)
            output = netC(inputs)
            if flag == 0:
                activation = torch.zeros(result_mid.size()[1]).to(args.device)
                flag = 1
            activation += torch.sum(result_mid, dim=[0, 2, 3])/len(data_set)
            # print("haha")
    hook.remove()
    # if args.device == 'cuda':
    #     netC.to('cpu')

    ### b. rank the mean of activation for each neural
    # Processing to get the "more important mask"
    # activation = torch.zeros(container[0].size()[1]).to(args.device)
    # for i in range(len(container)):
    #     activation +=  torch.sum(container[i], dim=[0, 2, 3])/len(data_set)
    # container = torch.cat(container, dim=0)
    # activation = torch.mean(container, dim=[0, 2, 3])
    if args.model == 'densenet161':
        out_channel = getattr(netC.features[-2],'denselayer24').conv2.out_channels
        seq_sort = torch.argsort(activation[-out_channel:])
    else:
        seq_sort = torch.argsort(activation)
    del container

    # import seaborn as sns
    # import matplotlib.pylab as plt
    #
    # activation_plot = activation.cpu().numpy().reshape((16, 32))
    # ax = sns.heatmap(activation_plot, linewidth=0.5, cmap='coolwarm', vmin=0, vmax=5)
    #
    # plt.title("2-D Heat Map")
    # plt.show()


    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    if args.model == 'preactresnet18':
        addtional_dim = 1
        pruning_mask_li = torch.ones(pruning_mask.shape[0] * addtional_dim, dtype=bool)
    if args.model == 'vgg16':
        addtional_dim = 49
        pruning_mask_li = torch.ones(pruning_mask.shape[0] * addtional_dim, dtype=bool)
    if args.model == 'vgg19':
        addtional_dim = 49
        pruning_mask_li = torch.ones(pruning_mask.shape[0] * addtional_dim, dtype=bool)
    if args.model == 'resnet18':
        addtional_dim = 1
        pruning_mask_li = torch.ones(pruning_mask.shape[0] * addtional_dim, dtype=bool)
        pruning_mask_li = pruning_mask_li.to(args.device)
    if args.model == 'densenet161':
        addtional_dim = netC.classifier.in_features - out_channel
        pruning_mask_li = torch.ones(netC.classifier.in_features, dtype=bool)
    if args.model == 'mobilenet_v3_large':
        addtional_dim = 1
        pruning_mask_li = torch.ones(pruning_mask.shape[0] * addtional_dim, dtype=bool)
    if args.model == 'efficientnet_b3':
        addtional_dim = 1
        pruning_mask_li = torch.ones(pruning_mask.shape[0] * addtional_dim, dtype=bool)
    

    ### c. according to the sorting results, prune and test the accuracy
    acc_dis = 0
    prune_result = []
    number_filters = []
    clean_accuracies = []
    ASRs = []
    # densenet_flag = False
    # Pruning times - no-tuning after pruning a channel!!!
    # Re-assigning weight to the pruned net
    for index in range(int(pruning_mask.shape[0])):
        net_pruned = copy.deepcopy(netC)
        num_pruned = index
        if index:
            channel = seq_sort[index - 1]
            pruning_mask[channel] = False
            test_data = torch.sum(pruning_mask)

            if args.model == 'densenet161':
                pruning_mask_li[channel+addtional_dim] = False
            else:
                pruning_mask_li[range(channel*addtional_dim, ((channel+1)*addtional_dim))] = False
        print("Pruned {} filters".format(num_pruned))
        if args.model == 'preactresnet18':
            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.linear = nn.Linear((pruning_mask.shape[0] - num_pruned)*addtional_dim, args.num_classes)
            for name, module in net_pruned._modules.items():
                if "layer4" == name:
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "linear" == name:
                    module.weight.data = netC.linear.weight.data[:, pruning_mask_li]
                    module.bias.data = netC.linear.bias.data
                else:
                    continue
        if args.model == 'vgg16':
            net_pruned.features[28] = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.classifier[0] = nn.Linear((pruning_mask.shape[0] - num_pruned)*addtional_dim, 4096)
            for name, module in net_pruned._modules.items():
                if "features" == name:
                    module[28].weight.data = netC.features[28].weight.data[pruning_mask]
                    module[28].ind = pruning_mask
                elif "classifier" == name:
                    module[0].weight.data = netC.classifier[0].weight.data[:, pruning_mask_li]
                    module[0].bias.data = netC.classifier[0].bias.data
                else:
                    continue
        if args.model == 'vgg19':
            net_pruned.features[34] = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.classifier[0] = nn.Linear((pruning_mask.shape[0] - num_pruned)*addtional_dim, 4096)
            for name, module in net_pruned._modules.items():
                if "features" == name:  
                    module[34].weight.data = netC.features[34].weight.data[pruning_mask]
                    module[34].ind = pruning_mask
                elif "classifier" == name:
                    module[0].weight.data = netC.classifier[0].weight.data[:, pruning_mask_li]
                    module[0].bias.data = netC.classifier[0].bias.data
                else:
                    continue
        if args.model == 'resnet18':
            net_pruned.layer4[0].conv1 = nn.Conv2d(
                256, pruning_mask.shape[0] - num_pruned, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
            net_pruned.layer4[0].bn1 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            net_pruned.layer4[0].conv2 = nn.Conv2d(
                pruning_mask.shape[0] - num_pruned, pruning_mask.shape[0] - num_pruned, (3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            net_pruned.layer4[0].bn2 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            net_pruned.layer4[0].shortcut[0] = nn.Conv2d(
                256, pruning_mask.shape[0] - num_pruned, (1, 1), stride=(2, 2), bias=False
            )
            net_pruned.layer4[0].shortcut[1] = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

            net_pruned.layer4[1].conv1 = nn.Conv2d(
                pruning_mask.shape[0] - num_pruned, pruning_mask.shape[0] - num_pruned, (3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            net_pruned.layer4[1].bn1 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0] - num_pruned, pruning_mask.shape[0] - num_pruned, (3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            net_pruned.layer4[1].bn2 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

            net_pruned.fc = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)
            # net_pruned.fc = MaskedLayer(pruning_mask.shape[0], 10, pruning_mask_li)
            for name, module in net_pruned._modules.items():
                if "layer4" == name:
                    # continue
                    module[0].conv1.weight.data = netC.layer4[0].conv1.weight.data[pruning_mask]
                    module[0].bn1.weight.data = netC.layer4[0].bn1.weight.data[pruning_mask]
                    module[0].bn1.bias.data = netC.layer4[0].bn1.bias.data[pruning_mask]
                    module[0].bn1.running_mean.data = netC.layer4[0].bn1.running_mean.data[pruning_mask]
                    module[0].bn1.running_var.data = netC.layer4[0].bn1.running_var.data[pruning_mask]
                    module[0].conv2.weight.data = netC.layer4[0].conv2.weight.data[pruning_mask][:, pruning_mask]
                    module[0].bn2.weight.data = netC.layer4[0].bn2.weight.data[pruning_mask]
                    module[0].bn2.bias.data = netC.layer4[0].bn2.bias.data[pruning_mask]
                    module[0].bn2.running_mean.data = netC.layer4[0].bn2.running_mean.data[pruning_mask]
                    module[0].bn2.running_var.data = netC.layer4[0].bn2.running_var.data[pruning_mask]
                    module[0].shortcut[0].weight.data = netC.layer4[0].shortcut[0].weight.data[pruning_mask]
                    module[0].shortcut[1].weight.data = netC.layer4[0].shortcut[1].weight.data[pruning_mask]
                    module[0].shortcut[1].bias.data = netC.layer4[0].shortcut[1].bias.data[pruning_mask]
                    module[0].shortcut[1].running_mean.data = netC.layer4[0].shortcut[1].running_mean.data[pruning_mask]
                    module[0].shortcut[1].running_var.data = netC.layer4[0].shortcut[1].running_var.data[pruning_mask]

                    module[1].conv1.weight.data = netC.layer4[1].conv1.weight.data[pruning_mask][:, pruning_mask]
                    module[1].bn1.weight.data = netC.layer4[1].bn1.weight.data[pruning_mask]
                    module[1].bn1.bias.data = netC.layer4[1].bn1.bias.data[pruning_mask]
                    module[1].bn1.running_mean.data = netC.layer4[1].bn1.running_mean.data[pruning_mask]
                    module[1].bn1.running_var.data = netC.layer4[1].bn1.running_var.data[pruning_mask]
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask][:, pruning_mask]
                    module[1].bn2.weight.data = netC.layer4[1].bn2.weight.data[pruning_mask]
                    module[1].bn2.bias.data = netC.layer4[1].bn2.bias.data[pruning_mask]
                    module[1].bn2.running_mean.data = netC.layer4[1].bn2.running_mean.data[pruning_mask]
                    module[1].bn2.running_var.data = netC.layer4[1].bn2.running_var.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "fc" == name:
                    module.weight.data = netC.fc.weight.data[:, pruning_mask_li]
                    module.bias.data = netC.fc.bias.data
                    # module.masked_fc.weight.data = netC.fc.weight.data
                    # module.masked_fc.bias.data = netC.fc.bias.data
                else:
                    continue
        if args.model == 'densenet161':
            # if index != 0:
            #     try:
            #         net_pruned = copy.deepcopy(net_pruned_now)
            #     except:
            #         logging.info('have no pruned net')
            #     if channel+1 > 1056:
            #         densenet_flag = True
            #         now_layer = (channel+1 - 1056) // 48 + 1
            #         out_channels = getattr(net_pruned.features[-2],'denselayer{}'.format(now_layer)).conv2.out_channels
            #         getattr(net_pruned.features[-2],'denselayer{}'.format(now_layer)).conv2 = nn.Conv2d(192, out_channels - 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            #         start = (now_layer - 1)*48+1056
            #         mask = torch.ones(out_channels, dtype=bool)
            #         mask[int(sum(pruning_mask[start:(channel+1)]))] = False
            #         getattr(net_pruned.features[-2],'denselayer{}'.format(now_layer)).conv2.weight.data = getattr(net_pruned_now.features[-2],'denselayer{}'.format(now_layer)).conv2.weight.data[mask]
            #         try:
            #             has_pruned += 1
            #         except:
            #             has_pruned = 1
            #         logging.info('prune densenet {} layers'.format(has_pruned))
            #     # else:
            #     #     out_channels = net_pruned.features[-3].conv.out_channels
            #     #     net_pruned.features[-3].conv = nn.Conv2d(2112, out_channels - 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #         net_pruned.features[-1] = nn.BatchNorm2d(2208 - has_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #         mask_li = torch.ones(2208 - has_pruned + 1, dtype=bool)
            #         mask_li[int(sum(pruning_mask[0:(channel+1)]))] = False
            #         net_pruned.features[-1].weight.data = net_pruned_now.features[-1].weight.data[mask_li]
            #         net_pruned.features[-1].bias.data = net_pruned_now.features[-1].bias.data[mask_li]
            #         out_features = net_pruned.classifier.out_features
            #         net_pruned.classifier = nn.Linear(2208 - has_pruned, out_features)
            #         net_pruned.classifier.weight.data = net_pruned_now.classifier.weight.data[:,mask_li]
            #         net_pruned.classifier.bias.data = net_pruned_now.classifier.bias.data
            #     else:
            #         continue
            # net_pruned_now = copy.deepcopy(net_pruned)
            conv_old = getattr(netC.features[-2],'denselayer24').conv2 
            getattr(net_pruned.features[-2],'denselayer24').conv2 = nn.Conv2d(conv_old.in_channels, conv_old.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            getattr(net_pruned.features[-2],'denselayer24').conv2.weight.data = conv_old.weight.data[pruning_mask]
            bn_old = netC.features[-1]
            net_pruned.features[-1] = nn.BatchNorm2d(bn_old.num_features - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            net_pruned.features[-1].weight.data = bn_old.weight.data[pruning_mask_li]
            net_pruned.features[-1].bias.data = bn_old.bias.data[pruning_mask_li]
            lin_old = netC.classifier
            net_pruned.classifier = nn.Linear(lin_old.in_features - num_pruned, args.num_classes)
            net_pruned.classifier.weight.data = lin_old.weight.data[:,pruning_mask_li]
            net_pruned.classifier.bias.data = lin_old.bias.data
        if args.model == 'efficientnet_b3':
            net_pruned.features[-1][0] = nn.Conv2d(384, pruning_mask.shape[0] - num_pruned, kernel_size=(1, 1), stride=(1, 1), bias=False) 
            net_pruned.features[-1][1] = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            net_pruned.classifier[-1] = nn.Linear((pruning_mask.shape[0] - num_pruned)*addtional_dim, args.num_classes)
            for name, module in net_pruned._modules.items():
                if "features" == name:
                    module[-1][0].weight.data = netC.features[-1][0].weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                    module[-1][1].weight.data = netC.features[-1][1].weight.data[pruning_mask]
                    module[-1][1].bias.data = netC.features[-1][1].bias.data[pruning_mask]
                elif "classifier" == name:
                    module[-1].weight.data = netC.classifier[-1].weight.data[:, pruning_mask_li]
                    module[-1].bias.data = netC.classifier[-1].bias.data
                else:
                    continue
        if args.model == 'mobilenet_v3_large':
            net_pruned.features[-1][0] = nn.Conv2d(160, pruning_mask.shape[0] - num_pruned, kernel_size=(1, 1), stride=(1, 1), bias=False) 
            net_pruned.features[-1][1] = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            net_pruned.classifier[0] = nn.Linear((pruning_mask.shape[0] - num_pruned)*addtional_dim, 1280)
            for name, module in net_pruned._modules.items():
                if "features" == name:
                    module[-1][0].weight.data = netC.features[-1][0].weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                    module[-1][1].weight.data = netC.features[-1][1].weight.data[pruning_mask]
                    module[-1][1].bias.data = netC.features[-1][1].bias.data[pruning_mask]
                elif "classifier" == name:
                    module[0].weight.data = netC.classifier[0].weight.data[:, pruning_mask_li]
                    module[0].bias.data = netC.classifier[0].bias.data
                else:
                    continue

        
        net_pruned.to(args.device)
        #test_loss_1, test_acc_cl_1 = test_epoch(args, testloader_clean, netC, criterion, 0, 'clean')
        #test_loss_1, test_acc_bd_1 = test_epoch(args, testloader_bd, netC, criterion, 0, 'bd')
        test_loss, test_acc_cl = test_epoch(args, testloader_clean, net_pruned, criterion, 0, 'clean')
        test_loss, test_acc_bd = test_epoch(args, testloader_bd, net_pruned, criterion, 0, 'bd')
        # print('Acc Clean: {:.3f} | Acc Bd: {:.3f}'.format(test_acc_cl, test_acc_bd))
        logging.info('Pruned {} filters | Acc Clean: {:.3f} | Acc Bd: {:.3f}'.format(num_pruned, test_acc_cl, test_acc_bd))
        prune_result.append("%d %0.4f %0.4f\n" % (index, test_acc_cl, test_acc_bd))
        number_filters.append(index)
        clean_accuracies.append(test_acc_cl)
        ASRs.append(test_acc_bd)
        ### d. find the last model with reasonable ACC 
        if index == 0:
            test_acc_cl_ori = test_acc_cl
            test_acc_bd_ori = test_acc_bd
            last_net = copy.deepcopy(net_pruned)
            last_index = 0
        if abs(test_acc_cl - test_acc_cl_ori)/test_acc_cl_ori < args.acc_ratio:
            if abs(test_acc_cl - test_acc_cl_ori)/test_acc_cl_ori < args.acc_ratio:
                last_net = copy.deepcopy(net_pruned)
                last_index = index
        else:
            break
        if args.device == 'cuda':
            net_pruned.to('cpu')
        del net_pruned
        # densenet_flag = False
        
    file_name = os.path.join(os.getcwd() + args.checkpoint_save, 'pruning_result_fc500.txt')
    with open(file_name, "w") as f:
        f.write('No \t CleanACC \t PoisonACC \n')
        f.writelines(prune_result)

    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.plot(number_filters, clean_accuracies, label="clean accuracy")
    # plt.plot(number_filters, ASRs, label="ASR")
    # plt.title("pruning results of fc")
    # plt.legend()
    # figure_name = os.path.join(os.getcwd() + args.checkpoint_save, 'fc.png')
    # plt.savefig(figure_name)
    # plt.show()

    ### e. finetune the model with validation data
    
    optimizer = torch.optim.SGD(last_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer)
    elif args.lr_scheduler ==  'CosineAnnealingLR':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, T_max=100)
    criterion = torch.nn.CrossEntropyLoss() 
    
    best_acc = 0
    best_asr = 0
    clean_accuracies = []
    ASRs = []
    with torch.no_grad():
        last_net.eval()
        asr_acc = 0
        for i, (inputs, labels) in enumerate(testloader_bd):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = last_net(inputs)
            pre_label = torch.max(outputs, dim=1)[1]
            asr_acc += torch.sum(pre_label == labels) / len(testloader_bd.dataset)
        ASRs.append(asr_acc.item())

        clean_acc = 0
        for i, (inputs, labels) in enumerate(testloader_clean):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = last_net(inputs)
            pre_label = torch.max(outputs, dim=1)[1]
            clean_acc += torch.sum(pre_label == labels) / len(testloader_clean.dataset)
        clean_accuracies.append(clean_acc.item())

    for j in range(args.epochs):
        batch_loss = []
        for i, (inputs,labels) in enumerate(val_loader):  # type: ignore
            last_net.train()
            last_net.to(args.device)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = last_net(inputs)
            loss = criterion(outputs, labels)
            batch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        one_epoch_loss = sum(batch_loss)/len(batch_loss)
        if args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler.step(one_epoch_loss)
        elif args.lr_scheduler == 'CosineAnnealingLR':
            scheduler.step()
        with torch.no_grad():
            last_net.eval()
            asr_acc = 0
            for i, (inputs,labels) in enumerate(testloader_bd):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = last_net(inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                asr_acc += torch.sum(pre_label == labels)/len(testloader_bd.dataset)
            ASRs.append(asr_acc.item())
            
            clean_acc = 0
            for i, (inputs,labels) in enumerate(testloader_clean):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = last_net(inputs)
                pre_label = torch.max(outputs, dim=1)[1]
                clean_acc += torch.sum(pre_label == labels)/len(testloader_clean.dataset)
            clean_accuracies.append(clean_acc.item())

        if not (os.path.exists(os.getcwd() + f'{args.checkpoint_save}')):
            os.makedirs(os.getcwd() + f'{args.checkpoint_save}')
        if best_acc < clean_acc:
            best_acc = clean_acc
            best_asr = asr_acc
            torch.save(
            {
                'model_name':args.model,
                'index': last_index,
                'model': last_net.cpu().state_dict(),
                'asr': asr_acc,
                'acc': clean_acc
            },
            f'./{args.checkpoint_save}defense_result.pt'
            )
        logging.info(f'Epoch{j}: clean_acc:{clean_acc} asr:{asr_acc} best_acc:{best_acc} best_asr{best_asr}')

    logging.info('Best Test Acc: {:.3f}%'.format(best_acc * 100))
    logging.info('Best Test Asr: {:.3f}%'.format(best_asr * 100))

    result = {}
    result['model'] = last_net
    result['prune_index'] = last_index
    return result

if __name__ == '__main__':
    
    ### 1. basic setting: args
    args = get_args()
    with open(args.yaml_path, 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    if args.dataset == "mnist":
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "cifar100":
        args.num_classes = 100
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "celeba":
        args.num_classes = 8
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/fp/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save)
    if args.log is None:
        args.log = save_path + '/saved/fp/'
    else:
        #args.log_file_name = args.result_file + '_' + str(args.seed)
        args.log_file_name = args.result_file[(args.result_file.rfind('/') + 1):] + '_' + str(args.seed)
    if not (os.path.exists(os.getcwd() + args.log)):
        os.makedirs(os.getcwd() + args.log)
    args.save_path = save_path

    ### 2. attack result(model, train data, test data)
    print(os.getcwd() + save_path + '/attack_result.pt')
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    ### 3. fp defense:
    result_defense = fp(args, result, config)

    ### 4. test the result and get ASR, ACC, RC 
    result_defense['model'].eval()
    result_defense['model'].to(args.device)
    tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x, y))
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=True, pin_memory=True)

    asr_acc = 0
    for i, (inputs, labels) in enumerate(data_bd_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs, dim=1)[1]
        asr_acc += torch.sum(pre_label == labels)
    asr_acc = asr_acc/len(data_bd_test)

    tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x, y))
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=True, pin_memory=True)

    clean_acc = 0
    for i, (inputs, labels) in enumerate(data_clean_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs, dim=1)[1]
        clean_acc += torch.sum(pre_label == labels)
    clean_acc = clean_acc/len(data_clean_test)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    robust_acc = -1
    if 'original_targets' in result['bd_test']:
        y_ori = result['bd_test']['original_targets']
        if y_ori is not None:
            if len(y_ori) != len(x):
                y_idx = result['bd_test']['original_index']
                y = y_ori[y_idx]
            else :
                y = y_ori
            data_bd_test = list(zip(x,y))
            data_bd_testset = prepro_cls_DatasetBD(
                full_dataset_without_transform=data_bd_test,
                poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
                bd_image_pre_transform=None,
                bd_label_pre_transform=None,
                ori_image_transform_in_loading=tran,
                ori_label_transform_in_loading=None,
                add_details_in_preprocess=False,
            )
            data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
        
            robust_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = result_defense['model'](inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                robust_acc += torch.sum(pre_label == labels)
            robust_acc = robust_acc/len(data_bd_test)
        

    if not (os.path.exists(os.getcwd() + f'{save_path}/fp/')):
        os.makedirs(os.getcwd() + f'{save_path}/fp/')
    torch.save(
    {
        'model_name':args.model,
        'model': result_defense['model'].cpu().state_dict(),
        'index': result_defense['prune_index'],
        'asr': asr_acc,
        'acc': clean_acc,
        'ra': robust_acc
    },
    os.getcwd() + f'{save_path}/fp/defense_result.pt'
    )

    