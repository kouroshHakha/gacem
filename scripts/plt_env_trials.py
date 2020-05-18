from utils.hdf5 import load_dict_from_hdf5
from pathlib import Path
from gacem.viz.plot import plt_hist2D
import numpy as np
from tqdm import tqdm
from spinup.utils.test_policy import load_policy_and_env


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--low', '-lo', type=float, default=-5.0, help='low')
    parser.add_argument('--high', '-hi', type=float, default=5.0, help='high')
    args = parser.parse_args()

    path = Path(args.path)
    # env, _ = load_policy_and_env(path)
    # action_space = env.action_space

    trial_path = path / 'trials'
    fig_path = trial_path / 'figs'
    fig_path.mkdir(exist_ok=True)
    # act_range = np.stack([action_space.low, action_space.high], axis=-1)
    act_range = np.array([[args.low, args.high], [args.low, args.high]])
    for trial_fname in tqdm(trial_path.iterdir()):
        if trial_fname.is_file() and trial_fname.suffix == '.h5':
            trial_d = load_dict_from_hdf5(trial_fname)
            data = trial_d['act']
            if data.shape[-1] != 2:
                raise ValueError('more than 2 dimensions not supported')
            plt_hist2D(data, fpath=fig_path / f'{trial_fname.stem}.png', bins=100,
                       cmap='binary', range=act_range)