#!/usr/bin/env python
import argparse
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evolve neural networks')

    args = parser.parse_args()

    print(f'[START] {datetime.now().isoformat()}')

    print(f'[DONE] {datetime.now().isoformat()}')
