#!/usr/bin/env python
import argparse
import os
from datetime import datetime

from evolution import evolution


def run():
    evolution.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evolve neural networks')

    args = parser.parse_args()

    print(f'[START] {datetime.now().isoformat()}')

    run()

    print(f'[DONE] {datetime.now().isoformat()}')
