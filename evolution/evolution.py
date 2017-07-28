import os

import neat
import numpy as np
from neat import Population
from neat.nn.feed_forward import FeedForwardNetwork

from evolution.genome import EWTGenome


def fitness(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 1.0
        net = FeedForwardNetwork.create(genome, config)
        output = net.activate(np.random.uniform(-1.0, 0.0, [10]))
        pass


def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config.ini')
    config = neat.Config(EWTGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path)
    population = Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    winner = population.run(fitness, 200)
