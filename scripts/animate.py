import argparse

from utils.pdb import register_pdb_hook
from gacem.viz.animate import convert

register_pdb_hook()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path to search in')
    parser.add_argument('pattern', type=str, help='pattern to search for')
    parser.add_argument('-f', dest='file_name', type=str, default='animation',
                        help='Optional file_name for saving without .gif extension')

    _args = parser.parse_args()
    return _args

if __name__ == '__main__':
    _args = parse_arguments()
    # noinspection PyUnresolvedReferences
    convert(_args.path, _args.pattern, _args.file_name)