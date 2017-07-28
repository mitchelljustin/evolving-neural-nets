from random import choice, randint, randrange

from neat.attributes import FloatAttribute
from neat.genes import BaseGene
from neat.genome import DefaultGenome
from neat.graphs import creates_cycle

HIDDEN_LAYER_SIZES = [8, 8, 4, 4]

class ParamsGene(BaseGene):
    __gene_attributes__ = [
        FloatAttribute('learning_rate'),
        FloatAttribute('layer_transfer_decay'),
    ]

    def distance(self, other, config):
        return 0


class EWTGenome(DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.params_gene = ParamsGene(0)
        self.layers = []

    def configure_new(self, config):
        super().configure_new(config)
        self.layers.append(config.input_keys)
        for layer_size in HIDDEN_LAYER_SIZES:
            layer = []
            for i in range(layer_size):
                new_node_id = self.get_new_node_key()
                ng = self.create_node(config, new_node_id)
                self.nodes[new_node_id] = ng
                layer.append(new_node_id)
            self.layers.append(layer)
        self.layers.append(config.output_keys)

    def mutate_params(self, config):
        self.params_gene.mutate(config)

    def mutate(self, config):
        super().mutate(config)

    def mutate_add_connection(self, config):
        src_layer = randrange(0, len(self.layers) - 1)
        dest_layer = randrange(src_layer + 1, len(self.layers))
        in_node = choice(self.layers[src_layer])
        out_node = choice(self.layers[dest_layer])
        key = (in_node, out_node)
        if key in self.connections:
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.layers = genome1.layers


