import multiprocessing as mp

import numpy as np
import pinocchio as pin

from self_collision import build_link_capsules, world_capsules


class Sphere:
    """A spherical obstacle: a world-frame position and a radius."""

    def __init__(self, position, radius):
        self.position = np.asarray(position, dtype=float)
        self.radius = float(radius)


def point_segment_distance(p, a, b):
    """Closest distance from point p to line segment [a, b]."""
    d = b - a
    L2 = d @ d
    if L2 < 1e-12:
        return np.linalg.norm(p - a)
    t = np.clip((p - a) @ d / L2, 0.0, 1.0)
    closest = a + t * d
    return np.linalg.norm(p - closest)


def check_external_collision(model, data, q, capsules, spheres):
    """True if any link capsule overlaps any sphere obstacle at q.

    capsules: link bounding capsules from self_collision.build_link_capsules.
    spheres: list of Sphere obstacles, in the same world frame as the model.
    """
    world = world_capsules(model, data, q, capsules)
    for p1, p2, radius in world:
        for sphere in spheres:
            if point_segment_distance(sphere.position, p1, p2) < (radius + sphere.radius):
                return True
    return False


# --- parallel external-collision filtering -----------------------------------
# Same pattern as self_collision's parallel internal filter: pinocchio
# Model/Data can't be pickled across process boundaries, so each worker
# builds its own copy once (via the Pool initializer) and reuses it for
# every q it checks.
_worker_model = None
_worker_data = None
_worker_capsules = None
_worker_spheres = None


def _init_worker(urdf_path, package_dirs, spheres):
    global _worker_model, _worker_data, _worker_capsules, _worker_spheres
    _worker_model, collision_model, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=package_dirs)
    _worker_data = _worker_model.createData()
    _worker_capsules = build_link_capsules(collision_model)
    _worker_spheres = spheres


def _is_external_collision_free(q):
    return not check_external_collision(_worker_model, _worker_data, q, _worker_capsules, _worker_spheres)


def filter_external_collision_free_parallel(q_pool, urdf_path, spheres, package_dirs=".", n_workers=None):
    """Filter q_pool down to the configurations with no sphere obstacle collision,
    checking in parallel across processes since each check is independent.

    urdf_path and package_dirs have to match whatever built q_pool, since each
    worker rebuilds its own model and capsules rather than sharing pinocchio
    objects.

    Returns (q_pool_external_collision_free, n_kept).
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    with mp.Pool(
        n_workers, initializer=_init_worker,
        initargs=(urdf_path, package_dirs, spheres),
    ) as pool:
        keep_mask = np.array(pool.map(_is_external_collision_free, q_pool, chunksize=256))

    kept = q_pool[keep_mask]
    return kept, len(kept)
