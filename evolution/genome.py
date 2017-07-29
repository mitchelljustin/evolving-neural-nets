from random import choice, randrange, sample, random

from neat.attributes import FloatAttribute
from neat.genes import BaseGene, DefaultGeneConfig, DefaultNodeGene, DefaultConnectionGene
from neat.genome import DefaultGenome, DefaultGenomeConfig

HIDDEN_LAYER_SIZES = [16, 8, 8, 4, 4]


class LearningRateGene(BaseGene):
  __gene_attributes__ = [
    FloatAttribute('learning_rate'),
    FloatAttribute('layer_transfer_decay'),
  ]

  @classmethod
  def parse_config(cls, config, param_dict):
    return DefaultGeneConfig(cls.__gene_attributes__, param_dict)

  def distance(self, other, config):
    return 0


class EWTGenomeConfig(DefaultGenomeConfig):
  def __init__(self, params_dict):
    super().__init__(params_dict)
    param_defs = LearningRateGene.get_config_params()
    for param_def in param_defs:
      setattr(self, param_def.name, param_def.interpret(params_dict))


class EWTGenome(DefaultGenome):
  def __init__(self, key):
    super().__init__(key)
    self.learning_rate_gene = LearningRateGene(0)
    self.layers = []

  @classmethod
  def parse_config(cls, param_dict):
    param_dict['node_gene_type'] = DefaultNodeGene
    param_dict['connection_gene_type'] = DefaultConnectionGene
    return EWTGenomeConfig(param_dict)

  def configure_new(self, config):
    super().configure_new(config)
    self.learning_rate_gene.init_attributes(config)
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
    for i in range(1, len(self.layers)):
      prev_layer = self.layers[i - 1]
      cur_layer = self.layers[i]
      for dest_node in cur_layer:
        src_nodes = sample(prev_layer, 2)
        for src_node in src_nodes:
          self.add_connection(config, src_node, dest_node, random() * 2 - 1, True)

  def mutate_params(self, config):
    self.learning_rate_gene.mutate(config)

  def mutate(self, config):
    super().mutate(config)
    self.mutate_params(config)

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
    self.learning_rate_gene = genome1.learning_rate_gene.crossover(genome2.learning_rate_gene)
