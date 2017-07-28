#!/usr/bin/env python
import argparse
import os
from datetime import datetime

def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'evolution/neat-config.ini')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evolve neural networks')

    args = parser.parse_args()

    print(f'[START] {datetime.now().isoformat()}')

    run()

    print(f'[DONE] {datetime.now().isoformat()}')
