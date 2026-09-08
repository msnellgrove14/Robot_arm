import numpy as np

from redundancy_ik import sample_redundancy_circle


def build_q_pool(positions, rotations, model, m=36, elbow_signs=(1, -1)):
    """For each valid X = (position, rotation), sweep the redundancy circle with m
    knot points in phi over both elbow branches using the closed-form analytical
    IK, and flatten every resulting q into one pool. Those are the roadmap's
    nodes. No mapping back from q to X is kept, since the graph only needs the
    pool of valid configurations.

    positions is (N, 3) and rotations is (N, 3, 3), as loaded from
    data/valid_X.npz. Returns an array of shape (n_nodes, model.nq).
    """
    all_q = []
    for p, R in zip(positions, rotations):
        qs = sample_redundancy_circle(model, p, R, n_phi=m, elbow_signs=elbow_signs)
        all_q.extend(qs)
    return np.array(all_q) if all_q else np.zeros((0, model.nq))
