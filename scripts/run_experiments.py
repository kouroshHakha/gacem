
from typing import cast, Type
import argparse
from copy import deepcopy
from pathlib import Path
import multiprocessing
from multiprocessing import Process
import glob
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utils.pdb import register_pdb_hook
from utils.importlib import import_class
from utils.file import read_yaml, write_yaml, read_pickle

from optnet.alg.base import AlgBase


register_pdb_hook()

def plot_exp(df, x, y, loc, **kwargs):
    plt.close()
    sns.lineplot(x=x, y=y, data=df, **kwargs)
    plt.savefig(loc)

def read_ckpt(path_dir):
    if (path_dir / 'checkpoint.pickle').is_file():
        return read_pickle(path_dir / 'checkpoint.pickle')
    if (path_dir / 'checkpoint.tar').is_file():
        import torch
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        return torch.load(path_dir / 'checkpoint.tar', map_location=device)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('spec_fpath', type=str, help='template spec file path')
    parser.add_argument('--nseeds', '-ns', type=int, default=1, help='number of seeds')
    parser.add_argument('--update-resume', dest='ur', default=False, action='store_true',
                        help='If True loads the results from root_dir in spec_file and continues '
                             'training')
    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == '__main__':
    # wierd issue with matplotlib backend https://github.com/matplotlib/matplotlib/issues/8795
    multiprocessing.set_start_method('spawn')

    _args = parse_arguments()
    # noinspection PyUnresolvedReferences
    fpath = _args.spec_fpath

    specs = read_yaml(fpath)
    alg_cls_str = specs['alg_class']
    alg_cls = cast(Type[AlgBase], import_class(alg_cls_str))

    params = specs['params']
    root_dir = Path('data', f'{params["prefix"]}_{params["suffix"]}')
    specs['root_dir'] = str(root_dir)

    processes = []
    for seed_iter in range(_args.nseeds):
        spec_seed = deepcopy(specs)
        seed = (seed_iter + 1) * 10
        spec_seed['params']['seed'] = seed
        spec_seed['params']['prefix'] = f's{seed}'
        spec_seed['params']['suffix'] = ''

        if _args.ur:
            pattern = str(root_dir / f's{seed}')
            ret_paths = glob.glob(pattern)
            if ret_paths:
                if len(ret_paths) == 1:
                    fpath = str(Path(ret_paths[0], 'params.yaml'))
                    write_yaml(fpath, spec_seed)
                    alg = alg_cls(fpath, load=_args.ur,
                                  use_time_stamp=False)
                else:
                    raise ValueError(f'More than 1 path with pattern {pattern} was found')
            else:
                raise ValueError(f'No path with pattern {pattern} was found')

        else:
            alg = alg_cls(spec_dict=spec_seed, use_time_stamp=False)

        p = Process(target=alg.main)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('-'*40 + ' all finished')
    # post-process results
    # data_keys = ['sim_cnt', 'sample_cnt', 'avg_cost', 'n_sols_in_buffer',
    #              'top_20', 'top_40', 'top_60']
    # acc_list, ent_list = [], []
    #
    # data = []
    # for i, path_dir in enumerate(root_dir.iterdir()):
    #     if not path_dir.is_dir():
    #         continue
    #     ckpt_dict = read_ckpt(path_dir)
    #     perf = read_yaml(path_dir / 'performance.yaml')
    #     seed_data = np.stack([ckpt_dict[key] for key in data_keys] + \
    #                          [10 * (i + 1) * np.ones_like(ckpt_dict[data_keys[0]])], axis=0)
    #     data.append(seed_data)
    #     acc_list.append(perf['acc'])
    #     ent_list.append(perf['ent'])
    #
    # data = np.concatenate(data, axis=-1)
    # df = pd.DataFrame(data.T, columns=data_keys+['seed'])
    # df.to_hdf(root_dir / 'df.hdf5', 'df', mode='w')
    #
    # ## --------- plots
    # plot_exp(df, 'sample_cnt', 'avg_cost', root_dir / 'avg_cost_sample_cnt.png')
    # plot_exp(df, 'sample_cnt', 'sim_cnt', root_dir / 'sim_cnt_sample_cnt.png')
    # plot_exp(df, 'sample_cnt', 'n_sols_in_buffer', root_dir / 'nsols_sample_cnt.png')
    # plot_exp(df, 'sample_cnt', 'top_20', root_dir / 'top_20.png')
    # plot_exp(df, 'sample_cnt', 'top_40', root_dir / 'top_40.png')
    # plot_exp(df, 'sample_cnt', 'top_60', root_dir / 'top_60.png')
    # # plot_exp(df, 'sim_cnt', 'avg_cost', root_dir / 'avg_cost_sim_cnt.png', hue='seed')
    # # plot_exp(df, 'sim_cnt', 'n_sols_in_buffer', root_dir / 'nsols_sim_cnt.png', hue='seed')
    #
    # # avg performance
    # save_dict = dict(acc_avg=float(np.mean(acc_list)), ent_avg=float(np.mean(ent_list)))
    # write_yaml(root_dir / 'avg_performance.yaml',  save_dict)
    # print(f'avg_acc = {save_dict["acc_avg"]:.6f}, avg_ent = {save_dict["ent_avg"]:.6f}')
