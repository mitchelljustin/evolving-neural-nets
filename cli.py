#!/usr/bin/env python
import argparse

from evolution import evolution
from evolution.fitness import fitness_objective, fitness_novelty_search, fitness_nslc1, fitness_nslc2, \
  fitness_nslc3

fitness_fns = {
  'objective': fitness_objective,
  'NS': fitness_novelty_search,
  'NSLC1': fitness_nslc1,
  'NSLC2': fitness_nslc2,
  'NSLC3': fitness_nslc3,
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
    fitfunc_name=args.fitness_fn,
    transfer_weights=args.transfer_weights,
    render=not args.no_render,
  ).run()
