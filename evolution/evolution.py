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
from evolution.nslc import compute_nslc_scores
from evolution.weight_transfer import transfer_weights
from maze_module.play import App

_NO_RENDER = None
NUM_ROUNDS_FOR_WT = 3


class Evolution:
    def __init__(self, fitness_function, transfer_weights, render) -> None:
        self.transfer_weights = transfer_weights
        self.fitness_function = fitness_function
        self.render = render

    def run_genome(self, info):
        genome_id, genome, config = info
        app = App(render=False)
        start = time.time()
        net = FeedForwardNetwork.create(genome, config)
        app.reset()
        result = app.execute([net])[0]
        end = time.time()
        print('Genome {} in {:.2f}s'.format(genome_id, end - start))
        return result

    def run_generation(self, genomes, config):
        pool = mps.Pool(4)
        num_rounds = NUM_ROUNDS_FOR_WT if self.transfer_weights else 1
        for round_no in range(num_rounds):
            info = [(g_id, g, config) for g_id, g in genomes]
            results = np.array(pool.map(self.run_genome, info))
            fitness = self.fitness_function(results)
            best_genomes = list(fitness.argsort()[-5:])
            if self.transfer_weights and round_no < NUM_ROUNDS_FOR_WT - 1:
                for i, (g1_id, genome1) in enumerate(genomes):
                    g2_i = i
                    while g2_i == i:
                        g2_i = random.sample(best_genomes, 1)[0]
                    g2_id, genome2 = genomes[g2_i]
                    transfer_weights(genome1, genome2)
                    print("Transferred weights from {} to {} ({})".format(g2_id, g1_id, genome1.learning_rate_gene))
            else:
                for i, f in enumerate(fitness):
                    g_id, genome = genomes[i]
                    print("Genome {} scored {:.02f}, {} connections, {}".format(g_id, f, len(genome.connections),
                                                                                genome.learning_rate_gene))
                    genome.fitness = f
                if self.render:
                    app = App(True)
                    nets = []
                    for gi in best_genomes:
                        _, genome = genomes[gi]
                        app.reset()
                        nets.append(FeedForwardNetwork.create(genome, config))

                    app.execute(nets)

    def run(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'neat-config.ini')
        config = neat.Config(EWTGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        population = Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        winner = population.run(self.run_generation, 20)

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
