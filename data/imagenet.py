import torchvision
import numpy as np
import torch
import os
import pickle
from copy import deepcopy

from config import imagenet_root
from config import imagenet21k_root
from config import osr_split_dir

osr_split_save_dir = os.path.join(osr_split_dir, 'imagenet_osr_splits.pkl')


class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


def pad_to_longest(list1, list2):

    if len(list2) > len(list1):

        list1 = [None] * (len(list2) - len(list1)) + list1

    elif len(list1) > len(list2):

        list2 = [None] * (len(list1) - len(list2)) + list2

    else:

        pass

    return list1, list2


def get_imagenet_osr_class_splits(imagenet21k_class_to_idx, num_imagenet21k_classes=1000,
                                  imagenet_root=imagenet_root, imagenet21k_root=imagenet21k_root,
                                  osr_split='random', precomputed_split_dir=osr_split_save_dir):

    if osr_split == 'random':

        """
        Find which classes in ImageNet21k are not in Imagenet1k, and select some of these classes as open-set classes
        """
        imagenet1k_classes = os.listdir(os.path.join(imagenet_root, 'val'))
        imagenet21k_classes = os.listdir(os.path.join(imagenet21k_root, 'val'))

        # Find which classes in I21K are not in I1K
        disjoint_imagenet21k_classes = set(imagenet21k_classes) - set(imagenet1k_classes)
        disjoint_imagenet21k_classes = list(disjoint_imagenet21k_classes)

        # Randomly select a number of OSR classes from them (must be less than ~10k as only ~11k valid classes in I21K)
        np.random.seed(0)
        selected_osr_classes = np.random.choice(disjoint_imagenet21k_classes, replace=False, size=(num_imagenet21k_classes,))

        # Convert class names to class indices
        selected_osr_classes_class_indices = [imagenet21k_class_to_idx[cls_name] for cls_name in selected_osr_classes]

        return selected_osr_classes_class_indices

    elif osr_split in ('Easy', 'Hard'):

        split_to_key = {
            'Easy': 'easy_i21k_classes',
            'Hard': 'hard_i21k_classes'
        }

        # Load splits
        with open(precomputed_split_dir, 'rb') as handle:
            precomputed_info = pickle.load(handle)

        osr_wnids = precomputed_info[split_to_key[osr_split]]
        selected_osr_classes_class_indices = \
            [imagenet21k_class_to_idx[cls_name] for cls_name in osr_wnids]

        return selected_osr_classes_class_indices

    else:

        raise NotImplementedError


def subsample_dataset(dataset, idxs):

    dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=list(range(1000))):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

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


def get_image_net_datasets(train_transform, test_transform, train_classes=range(1000),
                       open_set_classes=range(1000), num_open_set_classes=1000,
                       balance_open_set_eval=False, split_train_val=True, seed=0,
                       osr_split='random'):

    np.random.seed(seed)

    print('No validation split option for ImageNet dataset...')
    print('ImageNet datasets use hardcoded OSR splits...')
    print('Loading ImageNet Train...')
    # Init train dataset and subsample training classes
    train_dataset_whole = ImageNetBase(root=os.path.join(imagenet_root, 'train'), transform=train_transform)

    print('Loading ImageNet Val...')
    # Get test set for known classes
    test_dataset_known = ImageNetBase(root=os.path.join(imagenet_root, 'val'), transform=test_transform)

    print('Loading ImageNet21K Val...')
    # Get testset for unknown classes
    test_dataset_unknown = ImageNetBase(root=os.path.join(imagenet21k_root, 'val'), transform=test_transform)
    # Select which classes are open set
    open_set_classes = get_imagenet_osr_class_splits(test_dataset_unknown.class_to_idx,
                                                     num_imagenet21k_classes=num_open_set_classes,
                                                     osr_split=osr_split)

    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    all_datasets = {
        'train': train_dataset_whole,
        'val': test_dataset_known,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets