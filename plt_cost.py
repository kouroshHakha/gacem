import matplotlib.pyplot as plt
import argparse
import torch
from pathlib import Path

def plot_cost(args):
    ckpt_fs = args.ckpt
    legends = args.l
    fig_name = args.fig_name

    ax: plt.Axes = plt.gca()
    for ckpt_f, legend in zip(ckpt_fs, legends):
        checkpoint_data = torch.load(ckpt_f)
        buffer = checkpoint_data['buffer']
        import pdb
        pdb.set_trace()
        avg_cost = checkpoint_data['avg_cost']
        ax.plot(avg_cost, label=legend)

    ax.set_ylabel('avg_cost')
    ax.set_xlabel('iteration')
    ax.legend()
    save_dir = Path('data', 'figures')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'{fig_name}.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', nargs='+', help='list of checkpoint files with buffer entry',
                        required=True)
    parser.add_argument('-l', nargs='+', help='legend label in the same order as ckp files',
                        required=True)
    parser.add_argument('-f', dest='fig_name', default='cost', help='name of the figure file')
    args = parser.parse_args()

    assert len(args.ckpt) == len(args.l), 'Length mismatch between ckpt and legends'

    plot_cost(args)