'''
This file implements the defense method called finetuning (ft), which is a standard fine-tuning that uses clean data to finetune the model.

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. ft defense:
        a. get some clean data
        b. retrain the backdoor model
    4. test the result and get ASR, ACC, RC 
'''

import argparse
import logging
import os
import random
import sys




sys.path.append('../')
sys.path.append(os.getcwd())
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np

#from utils import args
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random 
from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
#from utils.input_aware_utils import progress_bar
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

    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)

    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--seed', type=str, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')

    parser.add_argument('--yaml_path', type=str, default="./config/defense/ft/config.yaml", help='the path of yaml')

    #set the parameter for the ft defense
    parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')

    arg = parser.parse_args()

    print(arg)
    return arg


def fine_tuning(arg, model, scaler, optimizer, scheduler, criterion, epoch, trainloader, testloader_cl = None, testloader_bd = None):

    total_clean, total_clean_correct, train_loss = 0, 0, 0
    batch_loss = []
    for i, (inputs, labels) in enumerate(trainloader):
        model.train()
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        # with torch.cuda.amp.autocast(enabled=False):
        #     outputs = model(inputs)
        #     loss = criterion(outputs, labels.long())
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        batch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
        #progress_bar(i, len(trainloader), 'Epoch: %d | Loss: %.3f | Training Acc: %.3f%% (%d/%d)' % (epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
        #print('Epoch:{} | Loss: {:.3f} | Training Acc: {:.3f}%({}/{})'.format(epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
        #logging.info('Epoch:{} | Loss: {:.3f} | Training Acc: {:.3f}%({}/{})'.format(epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
    model.eval()

    clean_accuracy = 0
    ASR = 0
    if testloader_cl is not None:
        total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
        for i, (inputs, labels) in enumerate(testloader_cl):
            inputs, labels = inputs.to(arg.device), labels.to(arg.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
            total_clean_test += inputs.shape[0]
            avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
            clean_accuracy = avg_acc_clean
            #progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
        # print('Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))
        logging.info('Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))
                
    if testloader_bd is not None:
        total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
        for i, (inputs, labels) in enumerate(testloader_bd):
            inputs, labels = inputs.to(arg.device), labels.to(arg.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
            total_clean_test += inputs.shape[0]
            avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
            ASR = avg_acc_clean
            #progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
        # print('Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))
        logging.info('Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))
    one_epoch_loss = sum(batch_loss)/len(batch_loss)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler.step(one_epoch_loss)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler.step()
    return model, clean_accuracy, ASR


def ft(args, result, config):
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    # logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
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

    # Prepare model, optimizer, scheduler
    model = generate_cls_model(model_name=args.model, num_classes=args.num_classes)
    model.load_state_dict(result['model'])
    model.to(args.device)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    x = result['clean_train']['x']
    y = result['clean_train']['y']
    data_set = list(zip(x, y))
    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    trainloader = data_loader

    #model_new = model
    model_new = generate_cls_model(model_name=args.model, num_classes=args.num_classes)
    old_state = result['model']
    #old_state.pop('fc.weight')
    # old_state.pop('fc.bias')
    model_new.load_state_dict(old_state, strict=False)
    model_new.to(args.device)

    tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=True)
    x = result['clean_val']['x']
    y = result['clean_val']['y']
    data_all_length = len(y)
    ran_idx = choose_index(args, data_all_length)
    log_index = os.getcwd() + args.log + "index_val_"+str(args.ratio)+ "_" + args.log_file_name + ".txt"
    np.savetxt(log_index, ran_idx, fmt='%d')
    data_set_val = list(zip([x[ii] for ii in ran_idx], [y[ii] for ii in ran_idx]))
    #data_set_val = list(zip(x, y))
    data_set_valset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set_val,
        poison_idx=np.zeros(len(data_set_val)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    val_loader = torch.utils.data.DataLoader(data_set_valset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
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

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x,y))
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

    optimizer = torch.optim.SGD(model_new.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    clean_accuracies = []
    ASRs = []
    with torch.no_grad():
        model_new.eval()
        asr_acc = 0
        for i, (inputs, labels) in enumerate(data_bd_loader):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model_new(inputs)
            pre_label = torch.max(outputs, dim=1)[1]
            asr_acc += torch.sum(pre_label == labels) / len(data_bd_loader.dataset)
        original_ASR = asr_acc.item()*100

        clean_acc = 0
        for i, (inputs, labels) in enumerate(data_clean_loader):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model_new(inputs)
            pre_label = torch.max(outputs, dim=1)[1]
            clean_acc += torch.sum(pre_label == labels) / len(data_clean_loader.dataset)
        original_ACC = clean_acc.item()*100

    best_ACC = 0
    best_ASR = 0
    for i in range(args.epochs):
        model_new, clean_accuracy, ASR = fine_tuning(args, model_new, scaler, optimizer, scheduler, criterion, i, val_loader, testloader_cl=data_clean_loader, testloader_bd=data_bd_loader)
        clean_accuracies.append(clean_accuracy)
        ASRs.append(ASR)
        if best_ACC < clean_accuracy:
            best_ACC = clean_accuracy
            best_ASR = ASR

    print("original ACC is", original_ACC, "and original ASR is", original_ASR)

    logging.info('Best Test Acc: {:.3f}%'.format(best_ACC))
    logging.info('Best Test Asr: {:.3f}%'.format(best_ASR))

    result = {}
    result['model'] = model_new
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
        args.checkpoint_save = save_path + '/record/defence/ft/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/ft/'
    else:
        #args.log_file_name = args.result_file + '_' + str(args.seed)
        args.log_file_name = args.result_file[(args.result_file.rfind('/') + 1):] + '_' + str(args.seed)
    if not (os.path.exists(os.getcwd() + args.log)):
        os.makedirs(os.getcwd() + args.log)
    args.save_path = save_path

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    print("Continue training...")
    ### 3. ft defense:
    result_defense = ft(args, result, config)

    ### 4. test the result and get ASR, ACC, RC 
    result_defense['model'].eval()
    result_defense['model'].to(args.device)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
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

    asr_acc = 0
    for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs,dim=1)[1]
        asr_acc += torch.sum(pre_label == labels)
    asr_acc = asr_acc/len(data_bd_test)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x,y))
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

    clean_acc = 0
    for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs,dim=1)[1]
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

    if not (os.path.exists(os.getcwd() + f'{save_path}/ft/')):
        os.makedirs(os.getcwd() + f'{save_path}/ft/')
    torch.save(
    {
        'model_name':args.model,
        'model': result_defense['model'].cpu().state_dict(),
        'asr': asr_acc,
        'acc': clean_acc,
        'ra': robust_acc
    },
    os.getcwd() + f'{save_path}/ft/defense_result.pt'
    )
