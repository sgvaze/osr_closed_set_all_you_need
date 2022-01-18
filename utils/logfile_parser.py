import re
import pandas as pd
import os
import numpy as np

from matplotlib import pyplot as plt
pd.options.display.width = 0

rx_dict = {
    'model_dir': re.compile(r'model_dir=\'(.*?)\''),
    'dataset': re.compile(r'dataset=\'(.*?)\''),
    'loss': re.compile(r'loss=\'(.*?)\''),
    'cs': re.compile(r'cs=(.*?),'),
    'm': re.compile(r'rand_aug_m=([-+]?\d*)'),
    'n': re.compile(r'rand_aug_n=([-+]?\d*)'),
    'performance': re.compile("[-+]?\d*\.\d+|\d+"),
    'split_idx': re.compile(r'split_idx=(\d)'),
    'seed': re.compile(r'seed=(\d)'),
    'runtime': re.compile(r'Total elapsed time \(h:m:s\): (.*?)\n'),
    'label_smoothing': re.compile("label_smoothing=([-+]?\d*\.\d+|\d+)"),
    'lr': re.compile(" lr=(\d*\.\d+|\d+)")
    # 'oscr': re.compile("label_smoothing=([-+]?\d*\.\d+|\d+)")
}

save_root_dir = '/work/sagar/open_set_recognition/sweep_summary_files/ensemble_pkls'

def get_file(path):

    file = []
    with open(path, 'rt') as myfile:
        for myline in myfile:  # For each line, read to a string,
            file.append(myline)

    return file


def parse_arpl_out_file(path, rx_dict, root_dir=save_root_dir, save_name='test.pkl', save=True, verbose=True):

    file = get_file(path=path)

    models = []
    for i, line in enumerate(file):

        if line.find('Namespace') != -1:

            model = {}
            s = rx_dict['model_dir'].search(line).group(1)
            exp_id = s[s.find("("):s.find(")") + 1]
            model['exp_id'] = exp_id

            model['M'] = rx_dict['m'].search(line).group(1)
            model['N'] = rx_dict['n'].search(line).group(1)
            model['split_idx'] = rx_dict['split_idx'].search(line).group(1)
            model['seed'] = rx_dict['seed'].search(line).group(1)
            model['dataset'] = rx_dict['dataset'].search(line).group(1)
            model['loss'] = rx_dict['loss'].search(line).group(1)
            model['cs'] = rx_dict['cs'].search(line).group(1)
            model['lr'] = rx_dict['lr'].search(line).group(1)

            if rx_dict['label_smoothing'].search(line) is not None:
                model['label_smoothing'] = rx_dict['label_smoothing'].search(line).group(1)

        if line.find('Finished') != -1:
            line_ = file[i - 1]

            perfs = rx_dict['performance'].findall(line_)[:4]
            model['Acc'] = perfs[1]
            model['AUROC'] = perfs[2]
            model['OSCR'] = perfs[3]
            model['runtime'] = rx_dict['runtime'].search(line).group(1)

            models.append(model)

    data = pd.DataFrame(models)

    if verbose:
        print(data)

    if save:

        save_path = os.path.join(root_dir, save_name)
        data.to_pickle(save_path)

    else:

        return data


def parse_multiple_files(all_paths, rx_dict, root_dir=save_root_dir, save_name='test.pkl', verbose=True, save=False):

    all_data = []
    for path in all_paths:

        data = parse_arpl_out_file(path, rx_dict, save=False, verbose=False)
        data['fname'] = path.split('/')[-1]
        all_data.append(data)

    all_data = pd.concat(all_data)
    save_path = os.path.join(root_dir, save_name)

    if save:
        all_data.to_pickle(save_path)

    if verbose:
        print(all_data)

    return all_data


save_dir = '/work/sagar/open_set_recognition/sweep_summary_files/ensemble_pkls'
base_path = '/work/sagar/open_set_recognition/slurm_outputs/myLog-{}.out'
# base_path = '/work/sagar/open_set_recognition/dev_outputs/logfile_{}.out'

# all_paths = [base_path.format(i) for i in ['401331_3']]
all_paths = [base_path.format(i) for i in ['{}_{}'.format(401331, j) for j in [4]]]
# all_paths = [base_path.format(i) for i in [507, 508, 509, 510, 511]]

data = parse_multiple_files(all_paths, rx_dict, verbose=True, save=False, save_name='test.pkl')
print(f"Mean Acc: {np.mean(data['Acc'].values.astype('float')):.2f}")
print(f"Mean AUROC: {np.mean(data['AUROC'].values.astype('float')):.2f}")
print(f"Mean OSCR: {np.mean(data['OSCR'].values.astype('float')):.2f}")
print(len(data))

print(data['exp_id'].values)