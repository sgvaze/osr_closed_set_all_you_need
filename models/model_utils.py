from models.classifier32 import classifier32
from methods.ARPL.arpl_models.resnetABN import resnet50ABN
from methods.ARPL.arpl_models.arpl_models import classifier32ABN
from methods.ARPL.loss.ARPLoss import ARPLoss

from utils.utils import strip_state_dict
from config import imagenet_moco_path, places_supervised_path, places_moco_path, imagenet_supervised_path

import timm
import torch
import argparse

from functools import partial

class TimmResNetWrapper(torch.nn.Module):

    def __init__(self, resnet):

        super().__init__()
        self.resnet = resnet

    def forward(self, x, return_features=True, dummy_label=None):

        x = self.resnet.forward_features(x)
        embedding = self.resnet.global_pool(x)
        if self.resnet.drop_rate:
            embedding = torch.nn.functional.dropout(embedding, p=float(self.drop_rate), training=self.training)
        preds = self.resnet.fc(embedding)

        if return_features:
            return embedding, preds
        else:
            return preds


class Classifier32ARPLWrapper(torch.nn.Module):

    def __init__(self, base_model, loss_layer):

        super().__init__()

        self.base_model = base_model
        self.loss_layer = loss_layer

    def forward(self, imgs, return_feature=False):

        x, y = self.base_model(imgs, True)
        logits, _ = self.loss_layer(x, y)

        if return_feature:
            return x, logits
        else:
            return logits

    def load_state_dict(self, state_dict):

        """
        Override method to take list of state dicts for loss layer and criterion
        """

        base_model_state_dict, loss_layer_state_dict = [strip_state_dict(s) for s in state_dict]
        # base_model_state_dict = strip_state_dict(base_model_state_dict, strip_key='base_model.')

        self.base_model.load_state_dict(base_model_state_dict)
        self.loss_layer.load_state_dict(loss_layer_state_dict)

        self.base_model.eval()
        self.loss_layer.eval()


def transform_moco_state_dict_places(obj, num_classes, supervised=False):

    """
    Transforms state dict from Places pretraining here: https://github.com/nanxuanzhao/Good_transfer
    :param obj: Moco State Dict
    :param args: argsparse object with training classes
    :return: State dict compatable with standard resnet architecture
    """

    if supervised:

        new_model = obj
        new_model['fc.weight'] = torch.randn((len(args.train_classes), 2048))
        new_model['fc.bias'] = torch.randn((len(args.train_classes),))

    else:

        newmodel = {}
        for k, v in obj.items():

            if k.startswith("fc.2"):
                continue

            if k.startswith("fc.0"):
                k = k.replace("0.", "")
                if "weight" in k:
                    v = torch.randn((num_classes, v.size(1)))
                elif "bias" in k:
                    v = torch.randn((num_classes,))

            newmodel[k] = v

    return newmodel


def transform_moco_state_dict(obj, num_classes):

    """
    :param obj: Moco State Dict
    :param args: argsparse object with training classes
    :return: State dict compatable with standard resnet architecture
    """

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder_q."):
            continue
        old_k = k
        k = k.replace("module.encoder_q.", "")

        if k.startswith("fc.2"):
            continue

        if k.startswith("fc.0"):
            k = k.replace("0.", "")
            if "weight" in k:
                v = torch.randn((num_classes, v.size(1)))
            elif "bias" in k:
                v = torch.randn((num_classes,))

        newmodel[k] = v

    return newmodel


def transform_moco_state_dict_arpl_cs(obj, num_classes):

    """
    :param obj: Moco State Dict
    :param args: argsparse object with training classes
    :return: State dict compatable with standard resnet architecture
    """

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder_q."):
            continue
        old_k = k
        k = k.replace("module.encoder_q.", "")

        if k.startswith("fc.2"):
            continue

        if k.startswith("fc.0"):
            k = k.replace("0.", "")
            if "weight" in k:
                v = torch.randn((num_classes, v.size(1)))
            elif "bias" in k:
                v = torch.randn((num_classes,))

        newmodel[k] = v

    # For newmodel, change all batch norms from bnX.XXX --> bnX.bns.0.XXX
    #                                                   add bnX.bns.1.XXX with same params

    newmodel2 = {}
    for k, v in newmodel.items():

        if 'bn' in k:
            parts = k.split('.')
            if k.startswith('bn1'):

                newk1 = '.'.join([parts[0], 'bns', '0', parts[-1]])
                newk2 = '.'.join([parts[0], 'bns', '1', parts[-1]])

            else:

                idx = [i for i, x in enumerate(parts) if 'bn' in x]
                idx = idx[0] + 1
                newk1 = '.'.join([*parts[:idx], 'bns', '0', *parts[idx:]])
                newk2 = '.'.join([*parts[:idx], 'bns', '1', *parts[idx:]])

            newmodel2[newk1] = v
            newmodel2[newk2] = v


        elif 'downsample' in k:

            if 'downsample.0' in k:

                newmodel2[k] = v

            else:

                parts = k.split('.')
                idx = len(parts) - 1

                newk1 = '.'.join([*parts[:idx], 'bns', '0', *parts[idx:]])
                newk2 = '.'.join([*parts[:idx], 'bns', '1', *parts[idx:]])

                newmodel2[newk1] = v
                newmodel2[newk2] = v

        else:

            newmodel2[k] = v


    return newmodel2


def get_model(args, wrapper_class=None, evaluate=False, *args_, **kwargs):

    if args.model == 'timm_resnet50_pretrained':

         # Get model
        if args.cs:
            model = resnet50ABN(num_classes=len(args.train_classes), num_bns=2, first_layer_conv=7)
        else:
            model = timm.create_model('resnet50', num_classes=len(args.train_classes))

        # Get function to transform state_dict and state_dict path
        if args.resnet50_pretrain == 'imagenet_moco':
            pretrain_path = imagenet_moco_path
            state_dict_transform = transform_moco_state_dict
        elif args.resnet50_pretrain == 'imagenet':
            pretrain_path = imagenet_supervised_path
            state_dict_transform = partial(transform_moco_state_dict_places, supervised=False)
        elif args.resnet50_pretrain == 'places_moco':
            pretrain_path = places_moco_path
            state_dict_transform = partial(transform_moco_state_dict_places, supervised=False)
        elif args.resnet50_pretrain == 'places':
            pretrain_path = places_supervised_path
            state_dict_transform = partial(transform_moco_state_dict_places, supervised=True)
        elif args.resnet50_pretrain == 'imagenet_moco' and args.cs:
            pretrain_path = imagenet_moco_path
            state_dict_transform = transform_moco_state_dict_arpl_cs  # Note, not implemented for imagenet pretraining
        else:
            raise NotImplementedError

        # Load pretrain weights
        state_dict = torch.load(pretrain_path) if args.resnet50_pretrain != 'imagenet_moco' \
            else torch.load(pretrain_path)['state_dict']
        state_dict = strip_state_dict(state_dict, strip_key='module.')
        state_dict = state_dict_transform(state_dict, len(args.train_classes))

        model.load_state_dict(state_dict)

        # If loss is ARPLoss, bolt on loss layer to model
        if args.loss == 'ARPLoss':
            if evaluate:
                model = TimmResNetWrapper(model)

                loss_layer = ARPLoss(use_gpu=True, weight_pl=0.0, temp=1, num_classes=len(args.train_classes),
                                     feat_dim=2048, label_smoothing=0.9)

                model = Classifier32ARPLWrapper(base_model=model, loss_layer=loss_layer)

    elif args.model == 'classifier32':

        try:
            feat_dim = args.feat_dim
            cs = args.cs
        except:
            feat_dim = None
            cs = False

        model = classifier32(num_classes=len(args.train_classes), feat_dim=feat_dim)

        if args.loss == 'ARPLoss':
            if evaluate:
                if cs:
                    model = classifier32ABN(feat_dim=feat_dim, num_classes=len(args.train_classes))

                    loss_layer = ARPLoss(use_gpu=True, weight_pl=0.0, temp=1, num_classes=len(args.train_classes),
                                         feat_dim=args.feat_dim, label_smoothing=0.9)

                    model = Classifier32ARPLWrapper(base_model=model, loss_layer=loss_layer)

    elif args.model in ['wide_resnet50_2', 'efficientnet_b0', 'efficientnet_b7', 'dpn92', 'resnet50']:

        model = timm.create_model(args.model, num_classes=len(args.train_classes))

    else:

        raise NotImplementedError

    if wrapper_class is not None:
        model = wrapper_class(model, *args_, **kwargs)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument('--model', default='timm_resnet50_pretrained', type=str)
    parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
    parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
    parser.add_argument('--loss', type=str, default='ARPLoss')
    args = parser.parse_args()

    args.train_classes = (0, 1, 8, 9)
    model = get_model(args)
    x, y = model(torch.randn(64, 3, 32, 32), True)
    debug = True