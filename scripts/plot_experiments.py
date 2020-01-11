
import argparse
from pathlib import Path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utils.pdb import register_pdb_hook
from utils.file import read_yaml


register_pdb_hook()

def plot_exp(df, x, y, hue, loc=None, legends=None, ax=None, **lplot_kwargs):
    if ax is None:
        plt.close()
        ax = plt.gca()
    sns.lineplot(x=x, y=y, hue=hue, data=df, ax=ax, **lplot_kwargs)
    if legends:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=legends)
    if loc:
        plt.savefig(loc)

def plot_with_goal(df, x, y, hue, loc, legends, goal=None, **lplot_kwargs):
    plt.close()
    ax = plt.gca()
    plot_exp(df, x, y, hue, legends=legends, ax=ax, **lplot_kwargs)
    if goal:
        ax.axhline(goal, color='k', linestyle='--')
    plt.savefig(loc)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folders', nargs='*', type=str, help='root_folders of directories')
    parser.add_argument('--save-path', dest='save_path',
                        default='', type=str, help='directory to save plots')
    parser.add_argument('--legends', '-l', nargs='*', help='list of strings to be used as legends')
    parser.add_argument('-g', dest='goal', type=float, help='the goal of the objective (i.e. 20)')
    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == '__main__':
    _args = parse_arguments()

    root_dir = Path(_args.save_path).absolute()
    root_dir.mkdir(parents=True, exist_ok=True)

    perf_list, df_list = [], []
    keys = [Path(path).stem for path in _args.folders]
    for key, path_str in zip(keys, _args.folders):
        path = Path(path_str)
        perf_list.append(read_yaml(path / 'avg_performance.yaml'))
        df_tmp = pd.read_hdf(path / 'df.hdf5', 'df')
        df_tmp['name'] = key
        df_list.append(df_tmp)

    perf_df = pd.DataFrame(perf_list, index=keys)
    perf_df.to_excel(root_dir / 'perf_cmp.xlsx')
    print(perf_df)

    df = pd.concat(df_list, ignore_index=True, sort=False)
    plot_exp(df, 'sample_cnt', 'n_sols_in_buffer', 'name', root_dir / 'nsols_sample_cnt.png',
             _args.legends)
    plot_with_goal(df, 'sample_cnt', 'top_20', 'name', root_dir / 'top_20.png', _args.legends,
                   _args.goal, ci='sd')
    plot_with_goal(df, 'sample_cnt', 'top_40', 'name', root_dir / 'top_40.png', _args.legends,
                   _args.goal, ci='sd')
    plot_with_goal(df, 'sample_cnt', 'top_60', 'name', root_dir / 'top_60.png', _args.legends,
                   _args.goal, ci='sd')

