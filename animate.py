from typing import Union

import argparse
from pathlib import Path
from utils.pdb import register_pdb_hook
import moviepy.editor as mpy
register_pdb_hook()

def convert(folder: Union[str, Path], pattern: str, file_name: str, fps: int = 10):
    folder_path = Path(folder)

    if not folder_path.is_dir():
        raise ValueError(f'Path {str(folder_path)} should be a directory')

    named_images = []
    for filename in folder_path.glob(pattern):
        named_images.append(str(filename.absolute()))

    named_images = sorted(named_images, key=lambda x: int(Path(x).stem.split('_')[2]))
    clip = mpy.ImageSequenceClip(named_images, fps=fps)
    clip.write_gif(folder_path / f'{file_name}.gif', fps=fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path to search in')
    parser.add_argument('pattern', type=str, help='pattern to search for')
    parser.add_argument('-f', dest='file_name', type=str, default='animation',
                        help='Optional file_name for saving without .gif extension')

    args = parser.parse_args()

    convert(args.path, args.pattern, args.file_name)