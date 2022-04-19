# Open-Set Recognition: a Good Closed-Set Classifier is All You Need?
Code for our paper: [Open-Set Recognition: a Good Closed-Set Classifier is All You Need?](https://www.robots.ox.ac.uk/~vgg/research/osr/)

**We tackle open-set recognition:** the task of detecting if a test sample comes from an unseen class (which the model did not see during training). 
We find a simple baseline of training a regular closed-set classifier as well as possible, and using the 'maximum logit score' (MLS) as an open-set indicator, can achieve SoTA on a number of evaluations.
We also propose the Semantic Shift Benchmark for open-set recognition and related tasks.

![image](assets/main_fig.png)

## Contents
[:boom: 1. Updates](#updates)

[:globe_with_meridians: 2. The Semantic Shift Benchmark](#ssb)

[:running: 3. Running](#running)

[:chart_with_upwards_trend: 4. Hyper-parameters](#hyperparams)

[:clipboard: 5. Citation](#cite)

## <a name="updates"/> :boom: Updates

### Updates to paper since pre-print

* We have added Stanford Cars to the Semantic Shift Benchmark. We have also provided ImageNet splits to be compatible with the Winter '21 ImageNet release.
    * When available, we have combined 'Medium' and 'Hard' SSB splits into a single 'Hard' split.
* Improved results in Table 1. The strong baseline now beats SoTA on four of six open-set datasets! 
    * See [this issue](https://github.com/sgvaze/osr_closed_set_all_you_need/issues/7) (thanks @agaldran!).
* We have added results on OoD (out-of-distribution detection) benchmarks in Appendix F. 
    * We find the strong baseline is competetive with SoTA OoD methods on the standard benchmark suite.
* We noticed that a ViT model we thought was trained only on ImageNet-1K was actually pre-trained on ImageNet-21K. This perhaps explains the anomalous results in Figure 3a.
    * We took models from [here](https://github.com/Alibaba-MIIL/ImageNet21K). The repo lists the model as 'pretrained on Imagenet-1K', but inspection of the paper shows this means pre-trained on ImageNet-21K before fine-tuning on ImageNet-1K. 

### Repo updates (18/01/2022)
* Included pre-trained weights for the MLS baseline on TinyImageNet in `pretrained_weights`.
* Included `bash_scripts/osr_train.sh` to train models on all splits for a given dataset using the tuned hyper-parameters from the paper.

## <a name="ssb"/> :globe_with_meridians: The Semantic Shift Benchmark

Download instructions for the datasets in the SSB can be found at the links below. The folder `data/open_set_splits` contains pickle files with the class splits. For each dataset, `data` contains functions which return PyTorch datasets containing 'seen' and 'unseen' classes according to the SSB splits. For the FGVC datasets, the pickle files also include information on which unseen classes are most similar to which seen classes.

* [ImageNet-21K-P](https://github.com/Alibaba-MIIL/ImageNet21K),
 [CUB](https://drive.google.com/drive/folders/1kFzIqZL_pEBVR7Ca_8IKibfWoeZc3GT1),
[Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html),
[FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

Links for the legacy open-set datasets are also available at the links below:
* [MNIST](https://pytorch.org/vision/stable/datasets.html),
[SVHN](https://pytorch.org/vision/stable/datasets.html),
[CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html),
[TinyImageNet](https://github.com/rmccorm4/Tiny-Imagenet-200)

For TinyImageNet, you also need to run `create_val_img_folder` in `data/tinyimagenet.py` to create
a directory with the test data.


## <a name="running"/> :running: Running

### Dependencies

```
pip install -r requirements.txt
```

### Config

Set paths to datasets and pre-trained models (for fine-grained experiments) in ```config.py```

Set ```SAVE_DIR``` (logfile destination) and ```PYTHON``` (path to python interpreter) in ```bash_scripts``` scripts.

### Scripts

**Train models**: To train models on all splits on a specified dataset (using tuned hyper-parameters from the paper), run:

```
bash bash_scripts/osr_train.sh
```

**Evaluating models**: Models can be evaluated by editing `exp_ids` in `methods/tests/openset_test.py`. The experiment IDs are printed in the `Namespace`
at the top of each log file.

**Pre-trained models**: Pre-trained weights for the MLS baseline on the five TinyImageNet splits can be found in `pretrained_weights/`. The models should achieve an average of 84.2% accuracy on the test-sets of the closed-set classes (across the five splits) and an average 83.0% AUROC on the open-set detection task. Models are all [VGG32](https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/models/classifier32.py) and use [this](https://github.com/sgvaze/osr_closed_set_all_you_need/blob/154360f0c6e6bab018d3db7765d092bddbd17b26/data/augmentations/__init__.py#L114) image normalization at test-time with `image_size=64`.

## <a name="hyperparams"/> :chart_with_upwards_trend: Optimal Hyper-parameters:

We tuned label smoothing and RandAug hyper-parameters to optimise **closed-set** accuracy on a single random validation
split for each dataset. For other hyper-parameters (image size, batch size, learning rate) we took values from 
the open-set literature for the standard datasets (specifically, the ARPL paper) and values from the FGVC literature
for the proposed FGVC benchmarks.

**Cross-Entropy optimal hyper-parameters:**

| **Dataset**       | **Image Size** | **Learning Rate** | **RandAug N** | **RandAug M** | **Label Smoothing** | **Batch Size** |
|---------------|------------|---------------|-----------|-----------|-----------------|------------|
| MNIST         | 32         | 0.1           | 1         | 8         | 0.0             | 128        |
| SVHN          | 32         | 0.1           | 1         | 18        | 0.0             | 128        |
| CIFAR-10      | 32         | 0.1           | 1         | 6         | 0.0             | 128        |
| CIFAR + N     | 32         | 0.1           | 1         | 6         | 0.0             | 128        |
| TinyImageNet  | 64         | 0.01          | 1         | 9         | 0.9             | 128        |
| CUB           | 448        | 0.001         | 2         | 30        | 0.3             | 32         |
| FGVC-Aircraft | 448        | 0.001         | 2         | 15        | 0.2             | 32         |

**ARPL + CS optimal hyper-parameters:**

(Note the lower learning rate for TinyImageNet)

| **Dataset**       | **Image Size** | **Learning Rate** | **RandAug N** | **RandAug M** | **Label Smoothing** | **Batch Size** |
|---------------|------------|---------------|-----------|-----------|-----------------|------------|
| MNIST         | 32         | 0.1           | 1         | 8         | 0.0             | 128        |
| SVHN          | 32         | 0.1           | 1         | 18        | 0.0             | 128        |
| CIFAR10      | 32         | 0.1           | 1         | 15         | 0.0             | 128        |
| CIFAR + N     | 32         | 0.1           | 1         | 6         | 0.0             | 128        |
| TinyImageNet  | 64         | 0.001          | 1         | 9         | 0.9             | 128        |
| CUB           | 448        | 0.001         | 2         | 30        | 0.2             | 32         |
| FGVC-Aircraft | 448        | 0.001         | 2         | 18        | 0.1             | 32         |

### Other

This repo also contains other useful utilities, including:
 * ```utils/logfile_parser.py```: To directly parse ```stdout``` outputs for Accuracy / AUROC metrics
 * ```data/open_set_datasets.py```: A useful framework for easily splitting existing datasets into controllable open-set splits
  into ```train```, ```val```, ```test_known``` and ```test_unknown```. Note: ImageNet has not yet been integrated here.
 * ```utils/schedulers.py```: Implementation of Cosine Warm Restarts with linear rampup as a PyTorch learning rate scheduler
  
## <a name="cite"/> :clipboard: Citation

If you use this code in your research, please consider citing our paper:
```
@InProceedings{vaze2022openset,
               title={Open-Set Recognition: a Good Closed-Set Classifier is All You Need?},
               author={Sagar Vaze and Kai Han and Andrea Vedaldi and Andrew Zisserman},
               booktitle={International Conference on Learning Representations},
               year={2022}}
```

Furthermore, please also consider citing
 [Adversarial Reciprocal Points Learning for Open Set Recognition](https://github.com/iCGY96/ARPL), upon whose code we build this repo.
