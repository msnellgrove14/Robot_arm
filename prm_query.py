import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from redundancy_ik import sample_redundancy_circle
from self_collision import check_self_collision
from external_collision import check_external_collision
from prm_edges import edge_internal_collision_free, edge_collision_free


def dijkstra_path(csgraph, start_idx, goal_idx):
    """Shortest path between two nodes in a PRM graph, using scipy's Dijkstra.

    scipy has no option to stop early at the target, so this computes distances
    and predecessors from start_idx to every node, then walks the predecessors
    back from goal_idx to build the path.

    Returns (path, distance), or (None, np.inf) if there is no path.
    """
    dist, predecessors = dijkstra(csgraph, directed=False, indices=start_idx, return_predecessors=True)

    if not np.isfinite(dist[goal_idx]):
        return None, np.inf

    path = [goal_idx]
    current = goal_idx
    while current != start_idx:
        current = predecessors[current]
        path.append(current)
    path.reverse()

    return path, dist[goal_idx]


def shortcut_path(q_path, model, data, capsules, excluded_pairs=None, spheres=None, max_step=0.1):
    """Greedy forward shortcutting: prune a path down to the fewest waypoints that
    are still reachable by valid direct edges.

    A graph shortest path minimizes joint-space distance, which does not mean it
    is direct in task space, so this cleans up the jaggedness afterward. From
    each anchor it tries the farthest waypoint ahead first, keeps the farthest
    one that is valid, and falls back to the next waypoint if nothing farther
    works.

    Pass the same sphere list the roadmap was built with, otherwise a shortcut
    can cut straight through an obstacle. With spheres=None only self collision
    is checked.

    Returns an array with the same columns as q_path and fewer rows.
    """
    q_path = np.asarray(q_path)
    n = q_path.shape[0]

    pruned = [q_path[0]]
    anchor = 0
    while anchor < n - 1:
        farthest = anchor + 1
        for j in range(n - 1, anchor, -1):
            if spheres is not None:
                valid = edge_collision_free(
                    model, data, q_path[anchor], q_path[j], capsules, spheres, excluded_pairs,
                    max_step=max_step, check_endpoints=False,
                )
            else:
                valid = edge_internal_collision_free(
                    model, data, q_path[anchor], q_path[j], capsules, excluded_pairs,
                    max_step=max_step, check_endpoints=False,
                )
            if valid:
                farthest = j
                break
        pruned.append(q_path[farthest])
        anchor = farthest

    return np.array(pruned)


def embed_X_in_graph(X_list, prm_graph, q_pool, model, data, capsules, excluded_pairs=None, spheres=None, m=8, k=10, max_step=0.1):
    """Embed task-space poses X = (position, rotation) into a PRM graph as extra
    nodes.

    The graph and q_pool passed in are left alone. This builds a copy in memory
    with the new rows and columns appended. For each X it generates candidate q's
    with the analytical redundancy IK, keeps the ones that are collision free,
    and connects each survivor to its k nearest neighbors across the combined
    node set.

    Returns (augmented_graph, augmented_q_pool, node_indices_per_X). The last one
    is a list of lists, one per X, holding that X's node indices in the augmented
    pool, and it is empty for an X whose candidates all failed the collision
    checks.
    """
    N = q_pool.shape[0]

    new_qs = []
    node_indices_per_X = []
    for p, R in X_list:
        candidates = sample_redundancy_circle(model, p, R, n_phi=m, elbow_signs=(1, -1))
        idxs_for_this_X = []
        for q in candidates:
            if check_self_collision(model, data, q, capsules, excluded_pairs):
                continue
            if spheres is not None and check_external_collision(model, data, q, capsules, spheres):
                continue
            idxs_for_this_X.append(N + len(new_qs))
            new_qs.append(q)
        node_indices_per_X.append(idxs_for_this_X)

    if not new_qs:
        return prm_graph, q_pool, node_indices_per_X

    new_qs = np.array(new_qs)
    augmented_q_pool = np.vstack([q_pool, new_qs])
    M = new_qs.shape[0]

    tree = cKDTree(augmented_q_pool)
    dists, idxs = tree.query(new_qs, k=k + 1)  # query only from new nodes, against the combined set

    seen_pairs = set()
    rows, cols, weights = [], [], []
    for local_i in range(M):
        i = N + local_i
        for rank in range(k + 1):
            j = int(idxs[local_i, rank])
            if j == i:
                continue
            pair = (i, j) if i < j else (j, i)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            qi, qj = augmented_q_pool[i], augmented_q_pool[j]
            if spheres is not None:
                valid = edge_collision_free(model, data, qi, qj, capsules, spheres, excluded_pairs, max_step=max_step, check_endpoints=False)
            else:
                valid = edge_internal_collision_free(model, data, qi, qj, capsules, excluded_pairs, max_step=max_step, check_endpoints=False)
            if valid:
                w = float(dists[local_i, rank])
                rows += [i, j]
                cols += [j, i]
                weights += [w, w]

    new_edges = coo_matrix((weights, (rows, cols)), shape=(N + M, N + M))
    old_coo = prm_graph.tocoo()
    augmented_graph = coo_matrix(
        (np.concatenate([old_coo.data, new_edges.data]),
         (np.concatenate([old_coo.row, new_edges.row]),
          np.concatenate([old_coo.col, new_edges.col]))),
        shape=(N + M, N + M),
    ).tocsr()

    return augmented_graph, augmented_q_pool, node_indices_per_X
