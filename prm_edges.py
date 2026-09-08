import numpy as np

from self_collision import check_self_collision
from external_collision import check_external_collision


def interpolate_q(q_a, q_b, max_step=0.1):
    """Linearly interpolate between two joint configs at enough resolution that no
    single joint moves more than max_step radians between consecutive points.
    That is the density a local planner needs to check collision and limits along
    a candidate edge without stepping past a thin obstacle.

    steps comes from the largest joint delta for that pair, so short edges get
    few points and long edges get proportionally more.

    Returns an array of shape (steps+1, nq), including both endpoints.
    """
    q_a = np.asarray(q_a)
    q_b = np.asarray(q_b)
    max_delta = np.max(np.abs(q_b - q_a))
    steps = max(1, int(np.ceil(max_delta / max_step)))
    alphas = np.linspace(0.0, 1.0, steps + 1)
    return q_a[None, :] + alphas[:, None] * (q_b - q_a)[None, :]


def edge_internal_collision_free(model, data, q_a, q_b, capsules, excluded_pairs=None, max_step=0.1, check_endpoints=True):
    """Can an edge be drawn between q_a and q_b as far as self collision goes? This
    interpolates between the two and checks every point on the path, returning
    True only if the whole path is clear.

    check_endpoints=False skips q_a and q_b themselves and only checks the
    interior points. Use it when the caller already knows both endpoints are
    collision free, such as nodes pulled from an already-filtered pool.
    """
    path = interpolate_q(q_a, q_b, max_step=max_step)
    points = path if check_endpoints else path[1:-1]
    return not any(check_self_collision(model, data, q, capsules, excluded_pairs) for q in points)


def edge_collision_free(model, data, q_a, q_b, capsules, spheres, excluded_pairs=None, max_step=0.1, check_endpoints=True):
    """Can an edge be drawn between q_a and q_b, checking both self collision and
    sphere obstacle collision along the interpolated path in one pass? An edge
    between two individually obstacle-free endpoints can still cut through an
    obstacle in between, so both checks run at every interpolated point.

    check_endpoints=False skips q_a and q_b themselves, the same as
    edge_internal_collision_free.
    """
    path = interpolate_q(q_a, q_b, max_step=max_step)
    points = path if check_endpoints else path[1:-1]
    for q in points:
        if check_self_collision(model, data, q, capsules, excluded_pairs):
            return False
        if check_external_collision(model, data, q, capsules, spheres):
            return False
    return True
