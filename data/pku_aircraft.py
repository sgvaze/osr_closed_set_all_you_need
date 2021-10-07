from torchvision.datasets import ImageFolder

import numpy as np
import os
from copy import deepcopy

from config import pku_air_root


class PKUAircraft(ImageFolder):

    def __init__(self, *args, **kwargs):

        super(PKUAircraft, self).__init__(*args, **kwargs)
        self.uq_idxs = np.arange(len(self))

    def __getitem__(self, item):

        img, label = super(PKUAircraft, self).__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


def subsample_dataset(dataset, idxs):

    dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def get_train_val_split(train_dataset, val_split=0.2):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    # Get training/validation datasets based on selected idxs
    train_dataset = subsample_dataset(train_dataset, train_idxs)
    val_dataset = subsample_dataset(val_dataset, val_idxs)

    return train_dataset, val_dataset


def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def get_pku_aircraft_datasets(train_transform, test_transform, train_classes=None,
                       open_set_classes=None, balance_open_set_eval=False, split_train_val=True, seed=0):

    # This dataset has only one 'known/unknown class' split

    np.random.seed(seed)

    train_path = os.path.join(pku_air_root, 'train')
    test_path = os.path.join(pku_air_root, 'test')
    out_path = os.path.join(pku_air_root, 'out')

    # Init train dataset and subsample training classes
    train_dataset_whole = PKUAircraft(root=train_path, transform=train_transform)

    # Split into training and validation sets
    if split_train_val:
        train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
        val_dataset_split.transform = test_transform
    else:
        train_dataset_split = train_dataset_whole
        val_dataset_split = None

    # Get test set for known classes
    test_dataset_known = PKUAircraft(root=test_path, transform=test_transform)

    # Get testset for unknown classes
    test_dataset_unknown = PKUAircraft(root=out_path, transform=test_transform)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {

        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,

    }

    return all_datasets


if __name__ == '__main__':

    datasets = get_pku_aircraft_datasets(None, None, balance_open_set_eval=False, split_train_val=False)
    print([len(v) for k,v in datasets.items()])