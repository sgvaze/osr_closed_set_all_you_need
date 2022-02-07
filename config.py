# ----------------------
# PROJECT ROOT DIR
# ----------------------
project_root_dir = '/users/sagar/open_world_learning/osr_closed_set_all_you_need/'

# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
exp_root = '/work/sagar/open_set_recognition'        # directory to store experiment output (checkpoints, logs, etc)
save_dir = '/work/sagar/open_set_recognition/methods/baseline/ensemble_entropy_test'    # Evaluation save dir

# evaluation model path (for openset_test.py and openset_test_fine_grained.py, {} reserved for different options)
root_model_path = '/work/sagar/open_set_recognition/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}.pth'
root_criterion_path = '/work/sagar/open_set_recognition/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}_criterion.pth'

# -----------------------
# DATASET ROOT DIRS
# -----------------------
cifar_10_root = '/work/sagar/datasets/cifar10'                                          # CIFAR10
cifar_100_root = '/work/sagar/datasets/cifar100'                                        # CIFAR100
cub_root = '/work/sagar/datasets/CUB'                                                   # CUB
aircraft_root = '/work/khan/datasets/aircraft/fgvc-aircraft-2013b'                      # FGVC-Aircraft
mnist_root = '/work/sagar/datasets/mnist/'                                              # MNIST
pku_air_root = '/work/sagar/datasets/pku-air-300/AIR'                                   # PKU-AIRCRAFT-300
car_root = "/work/sagar/datasets/stanford_car/cars_{}/"                                 # Stanford Cars
meta_default_path = "/work/sagar/datasets/stanford_car/devkit/cars_{}.mat"              # Stanford Cars Devkit
svhn_root = '/work/sagar/datasets/svhn'                                                 # SVHN
tin_train_root_dir = '/work/sagar/datasets/tinyimagenet/tiny-imagenet-200/train'        # TinyImageNet Train
tin_val_root_dir = '/work/sagar/datasets/tinyimagenet/tiny-imagenet-200/val/images'     # TinyImageNet Val
imagenet_root = '/scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12'              # ImageNet-1K
imagenet21k_root = '/work/sagar/datasets/imagenet21k_resized_new'                       # ImageNet-21K-P

# ----------------------
# FGVC / IMAGENET OSR SPLITS
# ----------------------
osr_split_dir = '/users/sagar/open_world_learning/osr_closed_set_all_you_need/data/open_set_splits'

# ----------------------
# PRETRAINED RESNET50 MODEL PATHS (For FGVC experiments)
# Weights can be downloaded from https://github.com/nanxuanzhao/Good_transfer
# ----------------------
imagenet_moco_path = '/work/sagar/pretrained_models/imagenet/moco_v2_800ep_pretrain.pth.tar'
places_moco_path = '/work/sagar/pretrained_models/places/moco_v2_places.pth'
places_supervised_path = '/work/sagar/pretrained_models/places/supervised_places.pth'
imagenet_supervised_path = '/work/sagar/pretrained_models/imagenet/supervised_imagenet.pth'