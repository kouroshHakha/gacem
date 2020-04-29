from typing import Union

from pathlib import Path


import moviepy.editor as mpy

def convert(folder: Union[str, Path], pattern: str, file_name: str, fps: int = 10):
    folder_path = Path(folder)

    if not folder_path.is_dir():
        raise ValueError(f'Path {str(folder_path)} should be a directory')

    named_images = []
    for filename in folder_path.glob(pattern):
        named_images.append(str(filename.absolute()))

    named_images = sorted(named_images, key=lambda x: int(Path(x).stem.split('_')[-1]))
    clip = mpy.ImageSequenceClip(named_images, fps=fps)
    clip.write_gif(folder_path / f'{file_name}.gif', fps=fps)

