import os
import torch
import random
import numpy as np
import inspect

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from typing import Optional
from torch.optim.optimizer import Optimizer

from datetime import datetime

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def strip_state_dict(state_dict, strip_key='module.'):

    """
    Strip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict


def init_experiment(args, runner_name=None):

    args.cuda = torch.cuda.is_available()

    if args.device == 'None':
        args.device = torch.device("cuda:0" if args.cuda else "cpu")
    else:
        args.device = torch.device(args.device if args.cuda else "cpu")

    print(args.gpus)

    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Unique identifier for experiment
    now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
          datetime.now().strftime("%S.%f")[:-3] + ')'

    log_dir = os.path.join(root_dir, 'log', now)
    while os.path.exists(log_dir):
        now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
              datetime.now().strftime("%S.%f")[:-3] + ')'

        log_dir = os.path.join(root_dir, 'log', now)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    os.mkdir(model_root_dir)

    args.model_dir = model_root_dir

    print(f'Experiment saved to: {args.log_dir}')

    args.writer = SummaryWriter(log_dir=args.log_dir)

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    args.writer.add_hparams(hparam_dict=hparam_dict, metric_dict={})

    print(runner_name)
    print(args)

    return args

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ClassificationPredSaver(object):

    def __init__(self, length, save_path=None):

        if save_path is not None:

            # Remove filetype from save_path
            save_path = save_path.split('.')[0]
            self.save_path = save_path

        self.length = length

        self.all_preds = None
        self.all_labels = None

        self.running_start_idx = 0

    def update(self, preds, labels=None):

        # Expect preds in shape B x C

        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()

        b, c = preds.shape

        if self.all_preds is None:
            self.all_preds = np.zeros((self.length, c))

        self.all_preds[self.running_start_idx: self.running_start_idx + b] = preds

        if labels is not None:
            if torch.is_tensor(labels):
                labels = labels.detach().cpu().numpy()

            if self.all_labels is None:
                self.all_labels = np.zeros((self.length,))

            self.all_labels[self.running_start_idx: self.running_start_idx + b] = labels

        # Maintain running index on dataset being evaluated
        self.running_start_idx += b

    def save(self):

        # Softmax over preds
        preds = torch.from_numpy(self.all_preds)
        preds = torch.nn.Softmax(dim=-1)(preds)
        self.all_preds = preds.numpy()

        pred_path = self.save_path + '.pth'
        print(f'Saving all predictions to {pred_path}')

        torch.save(self.all_preds, pred_path)

        if self.all_labels is not None:

            # Evaluate
            self.evaluate()
            torch.save(self.all_labels, self.save_path + '_labels.pth')

    def evaluate(self):

        topk = [1, 5, 10]
        topk = [k for k in topk if k < self.all_preds.shape[-1]]
        acc = accuracy(torch.from_numpy(self.all_preds), torch.from_numpy(self.all_labels), topk=topk)

        for k, a in zip(topk, acc):
            print(f'Top{k} Acc: {a.item()}')

def get_acc_auroc_curves(logdir):

    """
    :param logdir: Path to logs: E.g '/work/sagar/open_set_recognition/methods/ARPL/log/(12.03.2021_|_32.570)/'
    :return:
    """

    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    # Only gets scalars
    log_info = {}
    for tag in event_acc.Tags()['scalars']:

        log_info[tag] = np.array([[x.step, x.value] for x in event_acc.scalars._buckets[tag].items])

    return log_info
