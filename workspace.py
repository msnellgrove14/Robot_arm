import numpy as np
import pinocchio as pin

from inverse_kinematics import solve_ik


def max_reach(model):
    """Sum of the joint-to-joint offset magnitudes, which is how far the end
    effector reaches at full extension. Read off the model so it stays correct
    if the URDF changes.
    """
    total = 0.0
    for i in range(1, model.njoints):
        total += np.linalg.norm(model.jointPlacements[i].translation)
    return total


def in_task_space(xyz, model, safety_factor=0.9):
    """Rough reachability check that ignores joint limits and self collision. Is xyz
    inside the half sphere workspace approximation, with the floor at z=0 and the
    radius set to safety_factor times the arm's full reach?

    safety_factor shrinks the sphere so that sampled points stay clear of the
    fully outstretched singularity.

    xyz can be a single point of shape (3,), which returns a bool, or a batch of
    shape (N,3), which returns a boolean array.
    """
    xyz = np.asarray(xyz)
    radius = safety_factor * max_reach(model)
    above_floor = xyz[..., 2] >= 0
    within_radius = np.linalg.norm(xyz, axis=-1) <= radius
    return above_floor & within_radius


def is_pose_reachable(model, data, xyz, R, q_init=None, joint_id=7, **ik_kwargs):
    """Does IK actually converge to this position and orientation? This is a real
    Newton solve and is slow, so it is meant for confirming candidates that
    already passed in_task_space.
    """
    if q_init is None:
        q_init = pin.neutral(model)
    oMdes = pin.SE3(R, np.asarray(xyz))
    q, success, err, iters = solve_ik(model, data, joint_id, oMdes, q_init, **ik_kwargs)
    return success


def is_reachable(model, data, xyz, R, safety_factor=0.9, q_init=None, joint_id=7, **ik_kwargs):
    """Two-tier reachability check: run the cheap in_task_space filter first, then
    the IK-based is_pose_reachable on whatever passes.
    """
    if not in_task_space(xyz, model, safety_factor=safety_factor):
        return False
    return is_pose_reachable(model, data, xyz, R, q_init=q_init, joint_id=joint_id, **ik_kwargs)
