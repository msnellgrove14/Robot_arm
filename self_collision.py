import multiprocessing as mp

import numpy as np
import pinocchio as pin


def fit_link_capsule(vertices):
    """Bounding capsule, a line segment plus a radius, around a link's local-frame
    mesh vertices. The axis is the principal component from SVD, the endpoints
    are the vertex extent along that axis, and the radius is the largest distance
    from any vertex to the axis line. Every vertex is contained, since the
    rounded caps only make the capsule more conservative than a flat-capped
    cylinder.

    Returns (p1, p2, radius) in the same local frame as `vertices`.
    """
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]

    t = centered @ axis
    radial = centered - np.outer(t, axis)
    radius = np.linalg.norm(radial, axis=1).max()

    p1 = centroid + t.min() * axis
    p2 = centroid + t.max() * axis
    return p1, p2, radius


def build_link_capsules(collision_model):
    """Fit one bounding capsule per collision geometry, in that geometry's
    own local (parent-joint) frame. Returns a list of
    (parent_joint_id, p1, p2, radius), aligned with
    collision_model.geometryObjects.
    """
    return [
        (go.parentJoint, *fit_link_capsule(go.geometry.vertices()))
        for go in collision_model.geometryObjects
    ]


def segment_segment_distance(p1, p2, q1, q2):
    """Closest distance between 3D line segments [p1,p2] and [q1,q2].
    Standard closed-form segment-segment distance, clamped to each
    segment's parameter range (handles parallel/degenerate cases)."""
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r

    if a <= 1e-12 and e <= 1e-12:
        return np.linalg.norm(p1 - q1)
    if a <= 1e-12:
        t, s = 0.0, np.clip(f / e, 0.0, 1.0)
    elif e <= 1e-12:
        s, t = 0.0, np.clip(-(d1 @ r) / a, 0.0, 1.0)
    else:
        c = d1 @ r
        b = d1 @ d2
        denom = a * e - b * b
        t = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
        s = (b * t + f) / e
        if s < 0.0:
            s, t = 0.0, np.clip(-c / a, 0.0, 1.0)
        elif s > 1.0:
            s, t = 1.0, np.clip((b - c) / a, 0.0, 1.0)

    closest1 = p1 + t * d1
    closest2 = q1 + s * d2
    return np.linalg.norm(closest1 - closest2)


def world_capsules(model, data, q, capsules):
    """Transform each local-frame capsule to world frame at configuration q."""
    pin.forwardKinematics(model, data, q)
    return [
        (data.oMi[joint_id].act(p1), data.oMi[joint_id].act(p2), radius)
        for joint_id, p1, p2, radius in capsules
    ]


def default_excluded_pairs(n_links, window=2):
    """Kinematically-adjacent link-index pairs to skip -- their capsules
    are expected to touch/overlap right at the shared joint, which isn't a
    real self-collision."""
    return {
        (i, j) for i in range(n_links) for j in range(i + 1, n_links)
        if j - i <= window
    }


def check_self_collision(model, data, q, capsules, excluded_pairs=None):
    """True if any non-adjacent pair of link capsules overlaps at q."""
    if excluded_pairs is None:
        excluded_pairs = default_excluded_pairs(len(capsules))
    world = world_capsules(model, data, q, capsules)
    for i in range(len(world)):
        p1, p2, r1 = world[i]
        for j in range(i + 1, len(world)):
            if (i, j) in excluded_pairs:
                continue
            q1, q2, r2 = world[j]
            if segment_segment_distance(p1, p2, q1, q2) < (r1 + r2):
                return True
    return False


# --- parallel internal (self-)collision filtering ---------------------------
# Same pattern as prm_sampling's parallel IK verification: pinocchio
# Model/Data/GeometryModel can't be safely pickled across process
# boundaries, so each worker builds its own copy once (via the Pool
# initializer) and reuses it for every q it checks.
_worker_model = None
_worker_data = None
_worker_capsules = None
_worker_excluded_pairs = None


def _init_worker(urdf_path, package_dirs, window):
    global _worker_model, _worker_data, _worker_capsules, _worker_excluded_pairs
    _worker_model, collision_model, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=package_dirs)
    _worker_data = _worker_model.createData()
    _worker_capsules = build_link_capsules(collision_model)
    _worker_excluded_pairs = default_excluded_pairs(len(_worker_capsules), window=window)


def _is_internal_collision_free(q):
    return not check_self_collision(_worker_model, _worker_data, q, _worker_capsules, _worker_excluded_pairs)


def filter_internal_collision_free_parallel(q_pool, urdf_path, package_dirs=".", window=2, n_workers=None):
    """Filter q_pool down to the configurations with no self collision, checking in
    parallel across processes since each check is independent.

    urdf_path and package_dirs have to match whatever built the model q_pool came
    from, since each worker rebuilds its own model and collision model rather
    than sharing pinocchio objects.

    Returns (q_pool_internal_collision_free, n_kept).
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    with mp.Pool(
        n_workers, initializer=_init_worker,
        initargs=(urdf_path, package_dirs, window),
    ) as pool:
        keep_mask = np.array(pool.map(_is_internal_collision_free, q_pool, chunksize=256))

    kept = q_pool[keep_mask]
    return kept, len(kept)
