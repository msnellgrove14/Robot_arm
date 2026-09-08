import multiprocessing as mp

import numpy as np
import pinocchio as pin
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

from prm_edges import edge_internal_collision_free, edge_collision_free
from self_collision import build_link_capsules, default_excluded_pairs


def _knn_candidate_pairs(q_pool, k):
    """k-NN candidate edges over q_pool, deduped so a pair that shows up
    in either node's k-nearest list is only ever checked once. Fully
    vectorized (no Python-level loop over N*k candidates).

    Returns (lo, hi, weight) arrays: lo < hi are node indices, weight is
    the Euclidean distance between them.
    """
    N = q_pool.shape[0]
    tree = cKDTree(q_pool)
    dists, idxs = tree.query(q_pool, k=k + 1)

    row_i = np.repeat(np.arange(N), k)
    col_j = idxs[:, 1:].reshape(-1)
    weight = dists[:, 1:].reshape(-1)

    lo = np.minimum(row_i, col_j)
    hi = np.maximum(row_i, col_j)
    pair_key = lo.astype(np.int64) * N + hi.astype(np.int64)
    _, unique_idx = np.unique(pair_key, return_index=True)

    return lo[unique_idx], hi[unique_idx], weight[unique_idx]


def _pairs_to_graph(N, lo, hi, weight, keep_mask):
    lo_ok, hi_ok, w_ok = lo[keep_mask], hi[keep_mask], weight[keep_mask]
    rows = np.concatenate([lo_ok, hi_ok])
    cols = np.concatenate([hi_ok, lo_ok])
    weights = np.concatenate([w_ok, w_ok])
    return csr_matrix((weights, (rows, cols)), shape=(N, N))


def build_prm_graph(q_pool, model, data, capsules, excluded_pairs=None, k=10, max_step=0.1):
    """Build a k-nearest-neighbor PRM graph over q_pool. Candidate edges come from
    each node's k nearest neighbors by Euclidean distance in joint space, and are
    validated with edge_internal_collision_free. Every node in q_pool is assumed
    to already be internal-collision-free on its own, so the edge checks only
    validate the interior of each interpolated path.

    This is the serial reference version. See build_prm_graph_parallel for the
    one that scales to millions of nodes.

    Returns a symmetric weighted scipy.sparse.csr_matrix of shape (N, N), with
    the Euclidean distance between the two q's as the weight. No external
    collision or joint limit filtering happens here.
    """
    N = q_pool.shape[0]
    lo, hi, weight = _knn_candidate_pairs(q_pool, k)

    keep = np.array([
        edge_internal_collision_free(
            model, data, q_pool[i], q_pool[j], capsules, excluded_pairs,
            max_step=max_step, check_endpoints=False,
        )
        for i, j in zip(lo, hi)
    ])

    return _pairs_to_graph(N, lo, hi, weight, keep)


# --- parallel PRM graph build -----------------------------------------------
# Same pattern as the earlier parallel steps: pinocchio Model/Data can't be
# pickled across process boundaries, so each worker rebuilds its own copy
# once (via the Pool initializer). q_pool itself is also passed through the
# initializer rather than per-task, so only (i, j) index pairs cross the
# IPC boundary per task, not full q arrays.
_worker_model = None
_worker_data = None
_worker_capsules = None
_worker_excluded_pairs = None
_worker_max_step = None
_worker_q_pool = None


def _init_worker(urdf_path, package_dirs, window, max_step, q_pool):
    global _worker_model, _worker_data, _worker_capsules, _worker_excluded_pairs, _worker_max_step, _worker_q_pool
    _worker_model, collision_model, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=package_dirs)
    _worker_data = _worker_model.createData()
    _worker_capsules = build_link_capsules(collision_model)
    _worker_excluded_pairs = default_excluded_pairs(len(_worker_capsules), window=window)
    _worker_max_step = max_step
    _worker_q_pool = q_pool


def _check_edge_pair(pair):
    i, j = pair
    return edge_internal_collision_free(
        _worker_model, _worker_data, _worker_q_pool[i], _worker_q_pool[j],
        _worker_capsules, _worker_excluded_pairs,
        max_step=_worker_max_step, check_endpoints=False,
    )


def build_prm_graph_parallel(q_pool, urdf_path, package_dirs=".", k=10, max_step=0.1, window=2, n_workers=None):
    """Parallel version of build_prm_graph. Same vectorized k-NN candidate
    generation and dedup, but the candidate edges are validated across worker
    processes since each check is independent. q_pool is assumed to already be
    internal-collision-free per node.

    urdf_path and package_dirs have to match whatever built q_pool, since each
    worker rebuilds its own model and collision model instead of sharing
    pinocchio objects.

    Returns a symmetric weighted scipy.sparse.csr_matrix.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    N = q_pool.shape[0]
    lo, hi, weight = _knn_candidate_pairs(q_pool, k)
    pairs = list(zip(lo.tolist(), hi.tolist()))

    with mp.Pool(
        n_workers, initializer=_init_worker,
        initargs=(urdf_path, package_dirs, window, max_step, q_pool),
    ) as pool:
        keep = np.array(pool.map(_check_edge_pair, pairs, chunksize=2048))

    return _pairs_to_graph(N, lo, hi, weight, keep)


# --- parallel PRM graph build with external (sphere-obstacle) avoidance -----
_worker_spheres = None


def _init_worker_obstacles(urdf_path, package_dirs, window, max_step, q_pool, spheres):
    global _worker_model, _worker_data, _worker_capsules, _worker_excluded_pairs, _worker_max_step, _worker_q_pool, _worker_spheres
    _worker_model, collision_model, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=package_dirs)
    _worker_data = _worker_model.createData()
    _worker_capsules = build_link_capsules(collision_model)
    _worker_excluded_pairs = default_excluded_pairs(len(_worker_capsules), window=window)
    _worker_max_step = max_step
    _worker_q_pool = q_pool
    _worker_spheres = spheres


def _check_edge_pair_obstacles(pair):
    i, j = pair
    return edge_collision_free(
        _worker_model, _worker_data, _worker_q_pool[i], _worker_q_pool[j],
        _worker_capsules, _worker_spheres, _worker_excluded_pairs,
        max_step=_worker_max_step, check_endpoints=False,
    )


def build_prm_graph_with_obstacles_parallel(q_pool, urdf_path, spheres, package_dirs=".", k=10, max_step=0.1, window=2, n_workers=None):
    """Same as build_prm_graph_parallel, but validates candidate edges against both
    self collision and sphere obstacle collision using edge_collision_free.
    q_pool is assumed to already be collision free both ways.

    Returns a symmetric weighted scipy.sparse.csr_matrix.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    N = q_pool.shape[0]
    lo, hi, weight = _knn_candidate_pairs(q_pool, k)
    pairs = list(zip(lo.tolist(), hi.tolist()))

    with mp.Pool(
        n_workers, initializer=_init_worker_obstacles,
        initargs=(urdf_path, package_dirs, window, max_step, q_pool, spheres),
    ) as pool:
        keep = np.array(pool.map(_check_edge_pair_obstacles, pairs, chunksize=2048))

    return _pairs_to_graph(N, lo, hi, weight, keep)
