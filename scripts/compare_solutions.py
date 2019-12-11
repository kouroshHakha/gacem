from typing import cast, Type

from pathlib import Path
import argparse
import numpy as np

from optnet.data.vector import index_to_xval
from optnet.viz.plot import pca_scatter2D

from utils.pdb import register_pdb_hook
from utils.importlib import import_class
from utils.file import read_yaml, get_full_name
from utils.loggingBase import LoggingBase


register_pdb_hook()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('spec_fpath', type=str, help='spec file path')
    parsed_args = parser.parse_args()

    return parsed_args

def main(specs):
    nsamples = specs['nsamples']
    root_dir = Path(specs.get('root_dir', ''))

    prefix = specs.get('prefix', '')
    suffix = specs.get('suffix', '')

    samples_list, labels_list = [], []
    label_map = {}
    for label, (label_str, model_str) in enumerate(specs['models'].items()):
        print(f'sampling model {label} : {label_str}')
        model_path = root_dir / model_str / 'params.yaml'
        specs = read_yaml(model_path)
        alg_cls_str = specs.pop('alg_class')
        alg_cls = cast(Type[LoggingBase], import_class(alg_cls_str))
        alg = alg_cls(model_path, load=True)

        # noinspection PyUnresolvedReferences
        sample_ids = alg.load_and_sample_ids(nsamples)
        labels = np.ones(shape=sample_ids.shape[0]) * label
        label_map[label] = label_str
        # noinspection PyUnresolvedReferences
        samples_list.append(index_to_xval(alg.input_vectors, sample_ids))
        labels_list.append(labels)

    samples = np.concatenate(samples_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    name = get_full_name('comparison', prefix, suffix)
    save_path = root_dir / 'model_comparison' / f'{name}.png'
    pca_scatter2D(samples, labels, label_map, fpath=save_path, alpha=0.5,
              marker='o', edgecolors='none', s=10)

if __name__ == '__main__':
    _args = parse_arguments()

    # noinspection PyUnresolvedReferences
    fpath = _args.spec_fpath
    specs = read_yaml(fpath)

    main(specs)