import os

import neat

class BogusSpeciesSet(object):
    """ Encapsulates the default speciation scheme. """

    def __init__(self, config, reporters):
        pass

    @classmethod
    def parse_config(cls, param_dict):
        return {}

    @classmethod
    def write_config(cls, f, param_dict):
        pass

    def speciate(self, config, population, generation):
        pass

    def get_species_id(self, individual_id):
        return 0

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]

def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config.ini')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path)