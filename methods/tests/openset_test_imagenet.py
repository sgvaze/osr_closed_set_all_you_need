from test.utils import EvaluateOpenSet, ModelTemplate
from config import osr_split_dir

import torch
import pandas as pd
import torchvision
import argparse
import timm

from torch.utils.data import DataLoader
from data.imagenet import get_image_net_datasets
import sys, os

from torchvision import transforms
from torchvision.models import resnet, densenet, vgg


class MaxSoftmaxModel(ModelTemplate):

    def __init__(self, base_model, use_softmax=False):

        super(ModelTemplate, self).__init__()

        self.base_model = base_model
        self.use_softmax = use_softmax

    def forward(self, imgs):


        closed_set_preds = self.base_model(imgs)

        if self.use_softmax:
            closed_set_preds = torch.nn.Softmax(dim=-1)(closed_set_preds)

        open_set_preds = -closed_set_preds.max(dim=-1)[0]

        return closed_set_preds, open_set_preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
    parser.add_argument('--save_dir', type=str, default='/work/sagar/open_set_recognition/methods/baseline/'
                                                        'ensemble_entropy_test')
    parser.add_argument('--device', default='None', type=str, help='Which GPU to use')
    parser.add_argument('--osr_mode', default='max_softmax', type=str, help='{entropy, max_softmax}')
    parser.add_argument('--seed', default=0, type=int)

    # Data params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    # Train params
    args = parser.parse_args()

    # ------------------------
    # INIT
    # ------------------------
    device = torch.device('cuda:0')
    timm_config_path = os.path.join(osr_split_dir, 'timm_pretrained_accs.csv')

    timm_config = pd.read_csv(timm_config_path)
    timm_config = {row[1]['model']: row[1] for row in timm_config.iterrows()}

    # ------------------------
    # DEFINE TRANSFORMS
    # ------------------------
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


    def get_imagenet_standard_transform(image_size=224, crop_pct=0.875,
                                        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):

        test_transform_standard = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        return test_transform_standard

    # ------------------------
    # WHICH MODELS TO TEST
    # ------------------------
    # Models from TIMM
    all_models = ['resnet18', 'resnet50']   #['resnet18', 'resnet34', 'resnet50']

    # Models from the PyTorch library
    # torch_models = [
    #     'vgg11', 'vgg11_bn', 'vgg13'
    # ]
    # torch_models = [f'{x}_torch' for x in torch_models]
    # all_models = all_models + torch_models

    # ------------------------
    # NOTE: Following models have fault configs in timm
    # ------------------------
    # These models have the incorrect image sizes in their default configs
    faulty_config_models = [
        'resnet101d',
        'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
    ]

    for open_set_diff in ('Easy', 'Hard'):

        print(f'Starting eval for {open_set_diff} difficulty...')

        all_preds = {}

        # ------------------------
        # DATASETS AND DATALOADERS
        # ------------------------
        datasets = get_image_net_datasets(train_transform=None, test_transform=None, train_classes=range(1000),
                           open_set_classes=None, num_open_set_classes=1000, seed=0, osr_split=open_set_diff)


        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers)

        for model_name in all_models:

            # ------------------------
            # MODEL
            # ------------------------
            print(f'Doing inference on {model_name}...')

            # GET PYTORCH MODEL
            if 'torch' in model_name:

                resnet_model_name = model_name.split('_')[0]
                if 'vgg' in model_name:
                    base_model = vgg.__dict__[resnet_model_name](pretrained=True)
                elif 'resnet' in model_name:
                    base_model = resnet.__dict__[resnet_model_name](pretrained=True)
                elif 'densenet' in model_name:
                    base_model = densenet.__dict__[resnet_model_name](pretrained=True)
                else:
                    raise NotImplementedError

                # ------------------------
                # PREPARE TRANSFORMS
                # ------------------------
                image_size = 224
                crop_pct = 0.875
                interpolation = torchvision.transforms.InterpolationMode.BILINEAR
                mean = IMAGENET_DEFAULT_MEAN
                std = IMAGENET_DEFAULT_STD

                model_name = model_name.split('_')[0]  # So the code doesn't break later on

            # GET TIMM MODEL
            else:

                base_model = timm.create_model(model_name, pretrained=True, num_classes=1000).to(device)

                # ------------------------
                # PREPARE TRANSFORMS
                # ------------------------
                cfg = base_model.default_cfg
                image_size, interpolation, crop_pct, mean, std = \
                    cfg['input_size'], cfg['interpolation'], cfg['crop_pct'], cfg['mean'], cfg['std']

                assert interpolation in ('bilinear', 'bicubic')
                assert image_size[1] == image_size[2]
                image_size = image_size[1]

                interpolation = torchvision.transforms.InterpolationMode.BILINEAR \
                    if interpolation == 'bilinear' else torchvision.transforms.InterpolationMode.BICUBIC

                # Some models have the incorrect image size in the config:
                if model_name in faulty_config_models:
                    image_size = timm_config[model_name]['img_size']

            # Define model
            model = MaxSoftmaxModel(base_model=base_model, use_softmax=True)
            model.eval()
            model = model.to(device)

            # Get transform
            test_transform = get_imagenet_standard_transform(image_size=image_size, crop_pct=crop_pct,
                                                        interpolation=interpolation,
                                                        mean=mean, std=std)

            # Apply transform
            dataloaders['test_known'].dataset.transform = test_transform
            dataloaders['test_unknown'].dataset.transform = test_transform

            # ------------------------
            # EVALUATE
            # ------------------------

            evaluate = EvaluateOpenSet(model=model, known_data_loader=dataloaders['test_known'],
                                       unknown_data_loader=dataloaders['test_unknown'], device=device,
                                       save_dir=args.save_dir)

            # Make predictions on test sets
            evaluate.predict()
            preds = evaluate.evaluate(evaluate)
            all_preds[model_name] = preds
            print(preds)

            if model_name in timm_config.keys():
                print(f'TIMM Reported Closed-set Perf = {timm_config[model_name]["top1"]:.4f}')
            else:
                print(f'Model not in TIMM, look in https://pytorch.org/vision/stable/models.html for PyTorch model accs...')

        print('-----------------------------')
        for model_name, p in all_preds.items():

            test_acc, acc_95, auroc, aupr, oscr = p
            print(f'Open Set Difficulty: {open_set_diff}')
            print('{}: Acc: {:.3f} | AUROC: {:.3f} | OSCR: {:.3f}'.format(model_name, test_acc, auroc, oscr))

        print('-----------------------------')