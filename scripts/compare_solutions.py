from typing import cast, Type, Dict

from pathlib import Path
import argparse
import numpy as np
import itertools

from optnet.viz.plot import pca_scatter2d, tsne_scatter2d

from utils.pdb import register_pdb_hook
from utils.importlib import import_class
from utils.file import read_yaml, get_full_name, read_pickle, write_pickle
from utils.immutable import to_immutable
from utils.loggingBase import LoggingBase


register_pdb_hook()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('spec_fpath', type=str, help='spec file path')
    parser.add_argument('-f', '--force', default=False, action='store_true',
                        help='If True force regeneration of the plot, still loads the dataset if '
                             'necessary')
    parsed_args = parser.parse_args()

    return parsed_args

def main(specs, force_replot=False):
    nsamples = specs['nsamples']
    root_dir = Path(specs.get('root_dir', ''))

    prefix = specs.get('prefix', '')
    method = specs.get('method', 'pca')
    seed = specs.get('seed', 10)
    solution_only = specs.get('solution_only', False)

    samples_list, labels_list = [], []
    init_pop_list, pop_labels_list = [], []

    label_map = {}

    work_dir = root_dir / 'model_comparison'
    datasets_path = work_dir / 'datasets'
    datasets_path.parent.mkdir(exist_ok=True, parents=True)

    sol_all = 'sol' if solution_only else 'all'
    dataset_suf = f'n{nsamples}_' + sol_all
    fig_name = get_full_name('comparison', prefix, f'{method}_{sol_all}_s{seed}')

    # try reading the cache set
    try:
        cache = read_pickle(work_dir / 'cache.pickle')
    except FileNotFoundError:
        cache = set()

    # find a unique fname based on the content of spec file
    spec_immutable = to_immutable(specs)

    for index in itertools.count():
        fig_path = work_dir / f'{fig_name}_{index}.png'

        # increment index if fig_path exists and spec is new
        if not fig_path.exists() or force_replot:
            break
        else:
            if spec_immutable in cache:
                print('nothing is new')
                exit()

    cache.add(spec_immutable)

    # noinspection PyUnboundLocalVariable
    fig_title = str(fig_path.stem)

    for label, (label_str, model_str) in enumerate(specs['models'].items()):
        data_path = datasets_path / f'{model_str}_{dataset_suf}.pickle'

        if data_path.exists():
            print(f'loading dataset {label}: {label_str}')
            content = read_pickle(data_path)
            samples = content['samples']
        else:
            print(f'sampling model {label} : {label_str}')
            model_path = root_dir / model_str / 'params.yaml'
            model_specs = read_yaml(model_path)
            alg_cls_str = model_specs.pop('alg_class')
            alg_cls = cast(Type[LoggingBase], import_class(alg_cls_str))
            alg = alg_cls(model_path, load=True)

            # noinspection PyUnresolvedReferences
            samples = alg.load_and_sample(nsamples, only_positive=solution_only)
            print(f'saving into {str(data_path)}')
            write_pickle(data_path, dict(samples=samples))

        labels = np.ones(shape=samples.shape[0]) * label
        label_map[label] = label_str

        # content = read_pickle(root_dir / model_str / 'init_buffer.pickle')
        # init_pop = list(map(lambda x: x.item, content['init_buffer'].db_set.keys()))
        # init_pop_list += init_pop
        # pop_labels_list.append(np.ones(shape=len(init_pop)) * label)

        # noinspection PyUnresolvedReferences
        samples_list.append(samples)
        labels_list.append(labels)

    samples = np.concatenate(samples_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    # pops = np.stack(init_pop_list, axis=0)
    # pop_labels = np.concatenate(pop_labels_list, axis=0)

    if method == 'pca':
        pca_scatter2d(samples, labels, label_map, fpath=fig_path, alpha=0.5,
                      title=fig_title, edgecolors='none', s=10)
    elif method == 'tsne':
        # import matplotlib.pyplot as plt
        # plt.close()
        # _, axes = plt.subplots(2, 1)
        # tsne_scatter2d(samples, labels, label_map, seed=seed, ax=axes[0], alpha=0.5,
        #                title=fig_title, edgecolors='none', s=10)
        tsne_scatter2d(samples, labels, label_map, seed=seed, fpath=fig_path, alpha=0.5,
                       title=fig_title, edgecolors='none', s=10)
        # tsne_scatter2d(pops, pop_labels, label_map, seed=seed, ax=axes[1], alpha=0.5,
        #                title=fig_title, edgecolors='none', s=10)
        # plt.tight_layout()
        # plt.savefig(fig_path)
    else:
        raise ValueError('invalid dimensionality reduction, valid options are {"pca"| "tsne"}')

    # update cache
    write_pickle(work_dir / 'cache.pickle', cache)


if __name__ == '__main__':
    _args = parse_arguments()

    # noinspection PyUnresolvedReferences
    fpath = _args.spec_fpath
    force_replot = _args.force
    specs = read_yaml(fpath)

    main(specs, force_replot)
