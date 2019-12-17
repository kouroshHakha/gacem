import argparse
import subprocess
from pathlib import Path

SERVER_LIST = [
    Path('kourosh@pabti1.ist.berkeley.edu:~/projects/deep_ckt_workspace/autoreg_ckt/data'),
    Path('kourosh@pabti5.ist.berkeley.edu:~/projects/deep_ckt_workspace/autoreg_ckt/data')
]

def download(src: Path, index: int):
    subprocess.run(['scp', '-r', str(SERVER_LIST[index] / src), 'data'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs='*', help='list of folders in data that needs to be '
                                                   'downloaded')
    parser.add_argument('-i', '--index', type=int, default=0, help='index of the server')
    args = parser.parse_args()

    for _dir in args.folders:
        download(_dir, args.index)
