from test.utils import EvaluateOpenSet, ModelTemplate
from utils.utils import strip_state_dict

import torch
import argparse
import numpy as np
import pickle

from torch.utils.data import DataLoader
from data.open_set_datasets import get_datasets
from models.model_utils import get_model
import sys, os

from utils.utils import str2bool
from methods.ARPL.arpl_models.wrapper_classes import TimmResNetWrapper
from config import save_dir, osr_split_dir, root_model_path, root_criterion_path

class EnsembleModelEntropy(ModelTemplate):

    def __init__(self, all_models, mode='entropy', num_classes=4, use_softmax=False):

        super(ModelTemplate, self).__init__()

        self.all_models = all_models
        self.max_ent = torch.log(torch.Tensor([num_classes])).item()
        self.mode = mode
        self.use_softmax = use_softmax

    def entropy(self, preds):

        logp = torch.log(preds + 1e-5)
        entropy = torch.sum(-preds * logp, dim=-1)

        return entropy

    def forward(self, imgs):

        all_closed_set_preds = []

        for m in self.all_models:

            closed_set_preds = m(imgs, return_features=False)

            if self.use_softmax:
                closed_set_preds = torch.nn.Softmax(dim=-1)(closed_set_preds)

            all_closed_set_preds.append(closed_set_preds)

        closed_set_preds = torch.stack(all_closed_set_preds).mean(dim=0)

        if self.mode == 'entropy':
            open_set_preds = self.entropy(closed_set_preds)
        elif self.mode == 'max_softmax':
            open_set_preds = -closed_set_preds.max(dim=-1)[0]

        else:
            raise NotImplementedError

        return closed_set_preds, open_set_preds

def load_models(path, args):

    if args.loss == 'ARPLoss':

        model = get_model(args, wrapper_class=None, evaluate=True)
        state_dict_list = [torch.load(p) for p in path]
        model.load_state_dict(state_dict_list)

    else:

        model = get_model(args, wrapper_class=TimmResNetWrapper)
        state_dict = strip_state_dict(torch.load(path[0]))
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
    parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
    parser.add_argument('--device', default='None', type=str, help='Which GPU to use')
    parser.add_argument('--osr_mode', default='max_softmax', type=str, help='{entropy, max_softmax}')
    parser.add_argument('--seed', default=0, type=int)

    # Model
    parser.add_argument('--model', type=str, default='timm_resnet50_pretrained')
    parser.add_argument('--loss', type=str, default='Softmax')
    parser.add_argument('--feat_dim', default=2048, type=int)
    parser.add_argument('--max_epoch', default=599, type=int)
    parser.add_argument('--train_feat_extractor', default=True, type=str2bool,
                        help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
    parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)

    # Data params
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', type=str, default='scars')
    parser.add_argument('--transform', type=str, default='rand-augment')
    parser.add_argument('--exp_id', type=str, default='(19.05.2021_|_30.963)')

    # Train params
    args = parser.parse_args()
    args.save_dir = save_dir
    args.use_supervised_places = False

    device = torch.device('cuda:0')

    assert args.exp_id is not None

    # Define experiment IDs
    exp_ids = [
        args.exp_id,
    ]

    # Define paths
    all_paths_combined = [[x.format(i, args.dataset, args.dataset, args.max_epoch, args.loss)
                           for x in (root_model_path, root_criterion_path)] for i in exp_ids]

    all_preds = []

    # Get OSR splits
    osr_path = os.path.join(osr_split_dir, '{}_osr_splits.pkl'.format(args.dataset))

    with open(osr_path, 'rb') as f:
        class_info = pickle.load(f)

    train_classes = class_info['known_classes']
    open_set_classes = class_info['unknown_classes']


    for difficulty in ('Easy', 'Hard'):

        # ------------------------
        # DATASETS
        # ------------------------
        args.train_classes, args.open_set_classes = train_classes, open_set_classes[difficulty]

        if difficulty == 'Hard' and args.dataset != 'imagenet':
            args.open_set_classes += open_set_classes['Medium']

        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                image_size=args.image_size, balance_open_set_eval=False,
                                split_train_val=False, open_set_classes=args.open_set_classes)

        # ------------------------
        # DATALOADERS
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers)

        # ------------------------
        # MODEL
        # ------------------------
        print('Loading model...')
        all_models = [load_models(path=all_paths_combined[0], args=args)]

        model = EnsembleModelEntropy(all_models=all_models, mode=args.osr_mode, num_classes=len(args.train_classes))
        model.eval()
        model = model.to(device)

        # ------------------------
        # EVALUATE
        # ------------------------
        evaluate = EvaluateOpenSet(model=model, known_data_loader=dataloaders['test_known'],
                                   unknown_data_loader=dataloaders['test_unknown'], device=device, save_dir=args.save_dir)

        # Make predictions on test sets
        evaluate.predict()
        preds = evaluate.evaluate(evaluate, normalised_ap=False)
        all_preds.append(preds)

    all_preds = np.array(all_preds)
    means = np.mean(all_preds, axis=0)
    stds = np.std(all_preds, axis=0)
    print(f'Mean: {means[0]:.4f} pm {stds[0]:.4f} | AUROC: {means[-2]:.4f} pm {stds[-2]:.4f} | AUPR: {means[-1]:.4f} pm {stds[-1]:.4f}')
    print(all_preds)