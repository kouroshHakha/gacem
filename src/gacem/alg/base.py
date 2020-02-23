"""
This module implements a base structure for the algorithm with specific methods to be overridden
"""
from typing import Optional, Mapping, Any

import time
import abc
from pathlib import Path


from utils.file import read_yaml, write_yaml, get_full_name
from utils.loggingBase import LoggingBase


class AlgBase(LoggingBase, abc.ABC):
    # noinspection PyUnusedLocal
    def __init__(self, spec_file: str = '', spec_dict: Optional[Mapping[str, Any]] = None,
                 load: bool = False, **kwargs) -> None:
        LoggingBase.__init__(self)

        if spec_file:
            specs = read_yaml(spec_file)
        else:
            specs = spec_dict

        self.specs = specs
        self.load = load

        if load:
            self.work_dir = Path(spec_file).parent
        else:
            unique_name = time.strftime('%Y%m%d%H%M%S')
            suffix = specs['params'].get('suffix', '')
            prefix = specs['params'].get('prefix', '')
            unique_name = get_full_name(unique_name, prefix, suffix)
            self.work_dir = Path(specs['root_dir']) / f'{unique_name}'
            write_yaml(self.work_dir / 'params.yaml', specs, mkdir=True)

    @abc.abstractmethod
    def main(self) -> None:
        pass
