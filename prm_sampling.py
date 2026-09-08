import multiprocessing as mp

import numpy as np
import pinocchio as pin
from scipy.stats import qmc

from workspace import in_task_space, max_reach
from inverse_kinematics import solve_ik


def sample_batch(n, model, rng=None):
    """One Latin hypercube batch of (position, orientation) task-space samples,
    filtered by the cheap in_task_space check only.

    Positions are sampled over the cube x,y in [-R, R], z in [0, R], where R is
    the arm's max reach. Orientations are sampled as four values in [-1, 1],
    normalized to a unit quaternion and turned into a rotation matrix.

    Returns (positions, rotations, n_accepted). n_accepted is at most n, since
    points outside the half sphere get dropped.
    """
    R = max_reach(model)
    sampler = qmc.LatinHypercube(d=7, rng=rng)
    raw = sampler.random(n)  # (n, 7), each column uniform in [0, 1)

    xyz = np.empty((n, 3))
    xyz[:, 0] = -R + 2 * R * raw[:, 0]
    xyz[:, 1] = -R + 2 * R * raw[:, 1]
    xyz[:, 2] = R * raw[:, 2]

    quat_raw = 2 * raw[:, 3:7] - 1
    norms = np.linalg.norm(quat_raw, axis=1)
    norms[norms < 1e-9] = 1.0  # guard the (vanishingly unlikely) zero-vector case
    quat_xyzw = quat_raw / norms[:, None]

    accepted = in_task_space(xyz, model)
    xyz_acc = xyz[accepted]
    quat_acc = quat_xyzw[accepted]

    rotations = np.array([pin.Quaternion(q).matrix() for q in quat_acc])
    if len(rotations) == 0:
        rotations = rotations.reshape(0, 3, 3)

    return xyz_acc, rotations, len(xyz_acc)


def sample_valid_X(n, model, data=None, rng=None, joint_id=7, **ik_kwargs):
    """Draw n LHS samples, filter them with in_task_space, then check each survivor
    with a real IK solve so that only actually reachable poses come out.

    This is much slower than sample_batch because IK is a full Newton solve per
    point rather than a vectorized check.

    Returns (positions, rotations, q_solutions, n_valid).
    """
    if data is None:
        data = model.createData()

    xyz, rotations, n_cheap = sample_batch(n, model, rng=rng)

    valid_xyz = []
    valid_R = []
    valid_q = []
    for p, R in zip(xyz, rotations):
        oMdes = pin.SE3(R, p)
        q, success, err, iters = solve_ik(model, data, joint_id, oMdes, pin.neutral(model), **ik_kwargs)
        if success:
            valid_xyz.append(p)
            valid_R.append(R)
            valid_q.append(q)

    valid_xyz = np.array(valid_xyz) if valid_xyz else np.zeros((0, 3))
    valid_R = np.array(valid_R) if valid_R else np.zeros((0, 3, 3))
    valid_q = np.array(valid_q) if valid_q else np.zeros((0, model.nq))

    return valid_xyz, valid_R, valid_q, len(valid_xyz)


# --- parallel IK verification -----------------------------------------------
# pinocchio Model/Data can't be safely shared/pickled across process
# boundaries, so each worker builds its own copy once (via the Pool
# initializer) and reuses it for every task it handles, instead of
# rebuilding per-call.
_worker_model = None
_worker_data = None
_worker_joint_id = None
_worker_ik_kwargs = None


def _init_worker(urdf_path, package_dirs, joint_id, ik_kwargs):
    global _worker_model, _worker_data, _worker_joint_id, _worker_ik_kwargs
    _worker_model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=package_dirs)
    _worker_data = _worker_model.createData()
    _worker_joint_id = joint_id
    _worker_ik_kwargs = ik_kwargs


def _verify_one(args):
    p, R = args
    oMdes = pin.SE3(R, p)
    q, success, err, iters = solve_ik(
        _worker_model, _worker_data, _worker_joint_id, oMdes,
        pin.neutral(_worker_model), **_worker_ik_kwargs,
    )
    return p, R, q, success


def sample_valid_X_parallel(
    n, model, urdf_path, package_dirs=".", rng=None, joint_id=7,
    n_workers=None, **ik_kwargs,
):
    """Same as sample_valid_X, but runs the IK checks in parallel across processes.

    urdf_path and package_dirs have to match whatever built `model`, since each
    worker rebuilds its own model and data instead of sharing pinocchio objects.

    Returns (positions, rotations, q_solutions, n_valid).
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    xyz, rotations, n_cheap = sample_batch(n, model, rng=rng)
    if n_cheap == 0:
        return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, model.nq)), 0

    with mp.Pool(
        n_workers, initializer=_init_worker,
        initargs=(urdf_path, package_dirs, joint_id, ik_kwargs),
    ) as pool:
        results = pool.map(_verify_one, zip(xyz, rotations), chunksize=64)

    successes = [(p, R, q) for p, R, q, s in results if s]
    if successes:
        valid_xyz = np.array([p for p, R, q in successes])
        valid_R = np.array([R for p, R, q in successes])
        valid_q = np.array([q for p, R, q in successes])
    else:
        valid_xyz = np.zeros((0, 3))
        valid_R = np.zeros((0, 3, 3))
        valid_q = np.zeros((0, model.nq))

    return valid_xyz, valid_R, valid_q, len(valid_xyz)
