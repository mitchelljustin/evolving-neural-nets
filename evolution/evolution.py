import os
import random
import time
import multiprocessing as mps

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

def run_genome(info):
  genome_id, genome, config = info
  app = App(render=not _NO_RENDER)
  start = time.time()
  net = FeedForwardNetwork.create(genome, config)
  app.reset()
  result = app.on_execute(net)
  end = time.time()
  print('Genome {} in {:.2f}s'.format(genome_id, end - start))
  return result

def fitness_from_nslc(nslc: np.ndarray):
  return nslc[:, 0] * (2 ** ((nslc[:, 1]) / 3 - 2.5))

def run_generation(genomes, config):
  pool = mps.Pool(4)
  for round_no in range(NUM_ROUNDS):
    info = [(g_id, g, config) for g_id, g in genomes]
    results = np.array(pool.map(run_genome, info))
    nslc = nslc_fitness(results)
    fitness = fitness_from_nslc(nslc)
    if round_no < NUM_ROUNDS - 1:
      best_genomes = list(fitness.argsort()[-5:])
      for i, (g1_id, genome1) in enumerate(genomes):
        g2_i = i
        while g2_i == i:
          g2_i = random.sample(best_genomes, 1)[0]
        g2_id, genome2 = genomes[g2_i]
        learning_rate, layer_decay = transfer_weights(genome1, genome2)
        print("Transferred weights from {} to {} (lr={:.02e}, ld={:.02e}s)".format(g2_id, g1_id, learning_rate, layer_decay))
    else:
      for i, f in enumerate(fitness):
        g_id, genome = genomes[i]
        print("Genome {} scored {:.02f}, {} connections, {}".format(g_id, f, len(genome.connections), genome.learning_rate_gene))
        genome.fitness = f


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
  winner = population.run(run_generation, 300)

  node_names = {
    -1: 'laser left',
    -2: 'laser fwd left',
    -3: 'laser fwd',
    -4: 'laser fwd right',
    -5: 'laser right',
    -6: 'laser back',
    -7: 'pie left',
    -8: 'pie fwd',
    -9: 'pie right',
    -10: 'pie back',
    0: 'left/right',
    1: 'forward/backward',
  }
  if not os.path.exists('out'):
    os.mkdir('out')
  visualize.draw_net(config, winner, False, node_names=node_names, filename='out/network')
  visualize.plot_stats(stats, ylog=False, view=False)
