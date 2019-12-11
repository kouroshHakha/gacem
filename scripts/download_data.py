import argparse
import subprocess
from pathlib import Path

SERVER = Path('kourosh@pabti1.ist.berkeley.edu:~/projects/deep_ckt_workspace/autoreg_ckt/data')

def download(src: Path):
    subprocess.run(['scp', '-r', str(SERVER / src), 'data'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs='*', help='list of folders in data that needs to be '
                                                   'downloaded')
    args = parser.parse_args()

    for _dir in args.folders:
        download(_dir)
