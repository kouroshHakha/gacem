from typing import cast, Type

from pathlib import Path
import argparse
import numpy as np
import pickle

from optnet.viz.plot import pca_scatter2D, tsne_scatter2D

from utils.pdb import register_pdb_hook
from utils.importlib import import_class
from utils.file import read_yaml, get_full_name
from utils.loggingBase import LoggingBase


register_pdb_hook()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('spec_fpath', type=str, help='spec file path')
    # parser.add_argument('--load', default=False, action='store_true',
    #                     help='If True loads the dataset from model_comparison')
    parsed_args = parser.parse_args()

    return parsed_args

def main(specs):
    nsamples = specs['nsamples']
    root_dir = Path(specs.get('root_dir', ''))

    prefix = specs.get('prefix', '')
    suffix = specs.get('suffix', '')
    method = specs.get('method', 'pca')
    seed = specs.get('seed', 10)
    solution_only = specs.get('solution_only', False)

    samples_list, labels_list = [], []
    label_map = {}

    sol_all = 'sol' if solution_only else 'all'
    fig_name = get_full_name('comparison', prefix, f'{method}_{sol_all}_s{seed}')
    fig_path = root_dir / 'model_comparison' / f'{fig_name}.png'

    dataset_suf = f'n{nsamples}_' + sol_all
    dataset_name = get_full_name('dataset', prefix, dataset_suf)
    data_path = root_dir / 'model_comparison' / 'datasets' / f'{dataset_name}.pickle'
    data_path.parent.mkdir(exist_ok=True, parents=True)

    if data_path.exists():
        with open(data_path, 'rb') as f:
            ret_dict = pickle.load(f)
        samples = ret_dict['samples']
        labels = ret_dict['labels']
        label_map = ret_dict['label_map']
    else:
        for label, (label_str, model_str) in enumerate(specs['models'].items()):
            print(f'sampling model {label} : {label_str}')
            model_path = root_dir / model_str / 'params.yaml'
            specs = read_yaml(model_path)
            alg_cls_str = specs.pop('alg_class')
            alg_cls = cast(Type[LoggingBase], import_class(alg_cls_str))
            alg = alg_cls(model_path, load=True)

            # noinspection PyUnresolvedReferences
            samples = alg.load_and_sample(nsamples, only_positive=solution_only)
            labels = np.ones(shape=samples.shape[0]) * label
            label_map[label] = label_str
            # noinspection PyUnresolvedReferences
            samples_list.append(samples)
            labels_list.append(labels)

        samples = np.concatenate(samples_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        with open(data_path, 'wb') as f:
            pickle.dump(dict(samples=samples, labels=labels, label_map=label_map), f)

    fig_title = str(data_path.stem)
    if method == 'pca':
        pca_scatter2D(samples, labels, label_map, fpath=fig_path, alpha=0.5,
                      title=fig_title, marker='o', edgecolors='none', s=10)
    elif method == 'tsne':
        tsne_scatter2D(samples, labels, label_map, seed=seed, fpath=fig_path, alpha=0.5,
                       title=fig_title, marker='o', edgecolors='none', s=10)
    else:
        raise ValueError('invalid dimensionality reduction, valid options are {"pca"| "tsne"}')

if __name__ == '__main__':
    _args = parse_arguments()

    # noinspection PyUnresolvedReferences
    fpath = _args.spec_fpath
    specs = read_yaml(fpath)

    main(specs)