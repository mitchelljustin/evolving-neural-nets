from evolution.nslc import compute_nslc_scores


def fitness_objective(results):
    return results[:, -1]


def fitness_novelty_search(results):
    nslc_scores = compute_nslc_scores(results)
    return nslc_scores[:, 0]


def fitness_nslc1(results):
    damping_factor = 9.0
    nslc_scores = compute_nslc_scores(results)
    fitness = nslc_scores[:, 0] * (2 ** ((nslc_scores[:, 1]) / damping_factor))
    return fitness


def fitness_nslc2(results):
    damping_factor = 3.0
    nslc_scores = compute_nslc_scores(results)
    fitness = nslc_scores[:, 0] * (2 ** ((nslc_scores[:, 1]) / damping_factor))
    return fitness



def fitness_nslc3(results):
    damping_factor = 1.0
    nslc_scores = compute_nslc_scores(results)
    fitness = nslc_scores[:, 0] * (2 ** ((nslc_scores[:, 1]) / damping_factor))
    return fitness