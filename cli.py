#!/usr/bin/env python
import argparse

from evolution import evolution
from evolution.fitness import fitness_objective, fitness_novelty_search, fitness_nslc

fitness_fns = {
  'objective': fitness_objective,
  'NS': fitness_novelty_search,
  'NSLC1': fitness_nslc(9.0),
  'NSLC2': fitness_nslc(3.0),
  'NSLC3': fitness_nslc(1.0),
}

if __name__ == '__main__':
  parser = argparse.ArgumentParser('Evolve neural networks')
  parser.add_argument('-norender', dest='no_render', action='store_true')
  parser.add_argument('-fitness', dest='fitness_fn', default='objective')
  parser.add_argument('-wt', dest='transfer_weights', action='store_true')

  args = parser.parse_args()
  fitness_func = fitness_fns.get(args.fitness_fn, None)
  if fitness_func is None:
    raise Exception('Invalid fitness function')

  evolution.Evolution(
    fitness_function=fitness_func,
    transfer_weights=args.transfer_weights,
    render=not args.no_render,
  ).run()
