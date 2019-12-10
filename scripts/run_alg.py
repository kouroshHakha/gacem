
from typing import cast, Type
import argparse

from utils.pdb import register_pdb_hook
from utils.importlib import import_class
from utils.file import read_yaml
from utils.loggingBase import LoggingBase

register_pdb_hook()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('spec_fpath', type=str, default=10, help='class name')
    parser.add_argument('--load', default=False, action='store_true',
                        help='If True loads the results from root_dir in spec_file')
    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == '__main__':

    _args = parse_arguments()
    # noinspection PyUnresolvedReferences
    fpath = _args.spec_fpath

    specs = read_yaml(fpath)
    alg_cls_str = specs.pop('alg_class')
    alg_cls = cast(Type[LoggingBase], import_class(alg_cls_str))
    alg = alg_cls(_args.spec_fpath, load=_args.load)
    alg.main()
