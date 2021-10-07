import torch
from torch import nn
from models.model_utils import transform_moco_state_dict

from methods.ARPL.arpl_models.resnetABN import resnet50ABN
from methods.ARPL.arpl_models.ABN import MultiBatchNorm
import timm

class CIFARResNetWrapper(nn.Module):

    def __init__(self, resnet):

        super().__init__()
        self.resnet = resnet

    def forward(self, x, return_features=True, dummy_label=None):

        embedding = self.resnet.net(x)
        preds = self.resnet.fc(embedding)

        if return_features:
            return embedding, preds
        else:
            return preds


class TimmResNetWrapper(nn.Module):

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


def get_resnetABN_with_moco_weights(args):

    checkpoint = torch.load('/work/sagar/pretrained_models/imagenet/moco_v2_800ep_pretrain.pth.tar')
    state_dict = checkpoint['state_dict']
    moco_state_dict = transform_moco_state_dict(state_dict, num_classes=len(args.train_classes))

    new_state_dict = transform_moco_state_dict_to_ABN(moco_state_dict)

    r50_ABN = resnet50ABN(num_classes=len(args.train_classes), first_layer_conv=7)

    # <Most> keys in state_dict should match:
    #       Both ABN layers initialised with BN layers of MoCo network
    #       BN layers in downsampling blocks randomly initialised
    r50_ABN.load_state_dict(new_state_dict, strict=False)

    return r50_ABN

def transform_moco_state_dict_to_ABN(obj):

    """
    :param obj: Moco State Dict for regular Timm ResNet
    :param num_classes: num_classes
    :return: State dict compatable with standard ABN resnet architecture architecture
    """

    newmodel = {}
    for k, v in obj.items():

        old_k = k

        if "bn" in old_k:
            layer_name = old_k.split("bn")
            suffix = layer_name[-1][1:]
            bn_idx = layer_name[-1][0]
            prefix = layer_name[0]

            new_k_1 = prefix + 'bn{}.'.format(bn_idx) + "bns.0" + suffix
            new_k_2 = prefix + 'bn{}.'.format(bn_idx) + "bns.1" + suffix

            newmodel[new_k_1] = v
            newmodel[new_k_2] = v

        # if "downsample" in old_k:
        #
        #     layer_name = old_k.split("downsample")
        #     suffix = layer_name[-1].split('.')[-1]
        #     prefix = layer_name[0] + "downsample" + layer_name[-1][:3]
        #
        #     new_k_1 = prefix + 'bn{}.'.format(bn_idx) + "bns.0" + suffix
        #     new_k_2 = prefix + 'bn{}.'.format(bn_idx) + "bns.1" + suffix
        #
        #     newmodel[new_k_1] = v
        #     newmodel[new_k_2] = v

        # if k.startswith("fc.0"):
        #     k = k.replace("0.", "")
        #     if "weight" in k:
        #         v = torch.randn((num_classes, v.size(1)))
        #     elif "bias" in k:
        #         v = torch.randn((num_classes,))

        else:
            newmodel[k] = v

    return newmodel


if __name__ == '__main__':

    r50 = get_resnetABN_with_moco_weights()
    debug = True