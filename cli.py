#!/usr/bin/env python
import argparse
import os
from datetime import datetime

from evolution import evolution


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evolve neural networks')
    parser.add_argument('-norender', dest='no_render', action='store_true')

    args = parser.parse_args()

    evolution.run(no_render=args.no_render)
