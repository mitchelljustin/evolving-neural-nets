from evolution.nslc import compute_nslc_scores

def fitness_objective(results):
  return results[:, -1]

def fitness_novelty_search(results):
  nslc_scores = compute_nslc_scores(results)
  return nslc_scores[:, 0]

def fitness_nslc(damping_factor):
  def calc_fitness(results):
    nslc_scores = compute_nslc_scores(results)
    fitness = nslc_scores[:, 0] * (2 ** ((nslc_scores[:, 1]) / damping_factor))
    return fitness
  return calc_fitness

