import os
import random
import time

import neat
import numpy as np
from neat import Population
from neat.nn.feed_forward import FeedForwardNetwork

from evolution import visualize
from evolution.genome import EWTGenome
from evolution.nslc import nslc_fitness
from evolution.weight_transfer import transfer_weights
from maze_module.play import App

_NO_RENDER = None
NUM_ROUNDS = 3


def run_generation(genomes, config):
  app = App(render=not _NO_RENDER)
  results = np.zeros([len(genomes), 7])
  for round_no in range(NUM_ROUNDS):
    for i, (genome_id, genome) in enumerate(genomes):
      start = time.time()
      net = FeedForwardNetwork.create(genome, config)
      app.reset()
      result = app.on_execute(net)
      if round_no == NUM_ROUNDS - 1:
          results[i] = result
      end = time.time()
      print('Genome {} in {}'.format(genome_id, end - start))
    if round_no < NUM_ROUNDS - 1:
      for i, (g1_id, genome1) in enumerate(genomes):
          j = random.randrange(0, len(genomes) - 1)
          if j >= i:
              j += 1
          g2_id, genome2 = genomes[j]
          print("Transferring weights from {} to {}".format(g2_id, g1_id))
          transfer_weights(genome1, genome2)
  fitness = nslc_fitness(results)
  for i, (diversity, obj_rank) in enumerate(fitness):
      fitness = diversity * (2 ** (obj_rank / 3 - 2.5))
  g_id, genome = genomes[i]
  print("Genome {} scored {:.02f}".format(g_id, fitness))
  genome.fitness = fitness


def run(no_render=False):
  global _NO_RENDER
  _NO_RENDER = no_render
  local_dir = os.path.dirname(__file__)
  config_path = os.path.join(local_dir, 'neat-config.ini')
  config = neat.Config(EWTGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       config_path)
  population = Population(config)
  population.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  population.add_reporter(stats)
  winner = population.run(run_generation, 200)

  node_names = {
      -1:'laser left',
      -2: 'laser fwd left',
      -3: 'laser fwd',
      -4: 'laser fwd right',
      -5: 'laser right',
      -6: 'laser back',
      -7: 'pie left',
      -8: 'pie fwd',
      -9: 'pie right',
      -10: 'pie back',
      0:'left/right',
      1:'forward/backward',
  }
  if not os.path.exists('out'):
      os.mkdir('out')
  visualize.draw_net(config, winner, False, node_names=node_names, filename='out/network')
  visualize.plot_stats(stats, ylog=False, view=False)
