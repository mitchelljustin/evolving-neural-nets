import os

import neat
import numpy as np
from neat import Population
from neat.nn.feed_forward import FeedForwardNetwork

from evolution import visualize
from evolution.genome import EWTGenome
from evolution.weight_transfer import transfer_weights


def fitness(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = len(genome.connections)
        net = FeedForwardNetwork.create(genome, config)
        output = net.activate(np.random.uniform(-1.0, 0.0, [10]))
    g1 = genomes[0][1]
    g2 = genomes[1][1]
    if g1.fitness > 15:
        transfer_weights(g1, g2)


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
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

