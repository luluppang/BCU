# idea: select model you use in training and the trainer (the warper for training process)

import sys, logging
sys.path.append('../../')

import torch 
import torchvision.models as models
from typing import Optional

from utils.trainer_cls import ModelTrainerCLS
try:
    from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b3
except:
    logging.warning("efficientnet_b0,b3 fails to import, plz update your torch and torchvision")
try:
    from torchvision.models import mobilenet_v3_large
except:
    logging.warning("mobilenet_v3_large fails to import, plz update your torch and torchvision")

#trainer is cls
def generate_cls_model(
    model_name: str = 'resnet18',
    num_classes: int = 10,
    **kwargs,
):
    '''
    # idea: aggregation block for selection of classifcation models
    :param model_name:
    :param num_classes:
    :return:
    '''


    if model_name == 'resnet18':    ### resnet18 for cifar10 and gtsrb
        from models.resnet18_DBD import resnet18
        net = resnet18(num_classes=num_classes, **kwargs)
    elif model_name == 'resnet18comp':     ### resnet18 for imagenet10
        from models.resnet_comp import resnet18
        net = resnet18(num_classes=num_classes, **kwargs)
    # elif model_name == 'stdresnet18':
    #     from torchvision.models.resnet import resnet18
    #     net = resnet18(num_classes=num_classes, **kwargs)
    # elif model_name == 'pretrresnet18':
    #     from torchvision.models.resnet import resnet18
    #     net = resnet18(pretrained=True, **kwargs)
    #     net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        from models.vgg import vgg16
        net = vgg16(num_classes=num_classes, **kwargs)
    elif model_name == 'vgg16bn':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes=num_classes, **kwargs)
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net

def generate_cls_trainer(
        model,
        attack_name: Optional[str] = None,
        amp: bool = False,
):
    '''
    # idea: The warpper of model, which use to receive training settings.
        You can add more options for more complicated backdoor attacks.

    :param model:
    :param attack_name:
    :return:
    '''

    trainer = ModelTrainerCLS(
        model=model,
        amp=amp,
    )

    return trainer