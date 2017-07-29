from collections import defaultdict

from evolution.genome import EWTGenome


def transfer_weights(dest_genome: EWTGenome, src_genome: EWTGenome):
    learning_rate = 2 ** dest_genome.learning_rate_gene.learning_rate
    layer_decay = 2 ** dest_genome.learning_rate_gene.layer_transfer_decay
    common_conns = set(dest_genome.connections.keys()).intersection(src_genome.connections.keys())
    incoming_conns = defaultdict(list)
    for start, end in common_conns:
        incoming_conns[end].append((start, end))
    for i, layer in enumerate(dest_genome.layers[1:]):
        transfer_rate = learning_rate * layer_decay ** i
        for node in layer:
            conns = incoming_conns[node]
            for key in conns:
                src_wt = src_genome.connections[key].weight
                dest_wt = dest_genome.connections[key].weight
                delta = transfer_rate * (src_wt - dest_wt)
                dest_genome.connections[key].weight = delta + dest_wt
    return learning_rate, layer_decay


