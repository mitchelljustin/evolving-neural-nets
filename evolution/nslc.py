import numpy as np
from scipy.spatial import distance

NUM_NEAREST_NEIGHBOURS = 15

def nslc_fitness(individuals: np.ndarray):
    '''

    :param individuals: Nx7 array where the first 6 columns describe behaviour, and the last one describes objective value
    :return: Nx2 array, first column is novelty, second column is local competition score
    '''
    points = individuals[:, :6]
    num_points = len(points)
    objective_values = individuals[:, 6]
    distances = np.infty * np.ones([num_points, num_points])
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if j <= i:
                continue
            distances[i][j] = distance.euclidean(p1, p2)
    for i in range(num_points):
        distances[i:, i] = distances[i, i:]
    nearest = np.zeros([num_points, NUM_NEAREST_NEIGHBOURS, 2])
    for i, point in enumerate(points):
        point_distances = distances[i, :]
        for j in range(NUM_NEAREST_NEIGHBOURS):
            neighbour = np.argmin(point_distances)
            nearest[i, j, :] = [neighbour, point_distances[neighbour]]
            point_distances[neighbour] = np.infty
    novelty_score = np.average(nearest[:, :, 1], axis=1)
    local_comp_score = np.zeros([num_points])
    for i in range(num_points):
        indices = nearest[i, :, 0].reshape([NUM_NEAREST_NEIGHBOURS]).astype(np.int64)
        comp_score = np.sum(np.less(np.take(objective_values, indices), objective_values[i]).astype(np.float32))
        local_comp_score[i] = comp_score
    total_score = np.stack([novelty_score, local_comp_score], -1)
    return total_score

if __name__ == '__main__':
    individuals = np.random.uniform(0.0, 200.0, [250, 7])
    print(nslc_fitness(individuals))