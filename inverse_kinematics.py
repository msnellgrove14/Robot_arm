import numpy as np
from numpy.linalg import norm
import pinocchio as pin


def solve_ik(
    model,
    data,
    joint_id,
    oMdes,
    q_init,
    eps=1e-7,
    IT_MAX=1000,
    DT=1e-1,
    damp=1e-2,
):
    """Damped-least-squares (Levenberg-Marquardt) Newton IK to a single SE3
    target. Returns (q, success, err, iters).
    """
    q = q_init.copy()
    i = 0
    while True:
        pin.forwardKinematics(model, data, q)
        iMd = data.oMi[joint_id].actInv(oMdes)
        err = pin.log(iMd).vector
        if norm(err) < eps:
            return q, True, err, i
        if i >= IT_MAX:
            return q, False, err, i

        J = pin.computeJointJacobian(model, data, q, joint_id)
        J = -np.dot(pin.Jlog6(iMd.inverse()), J)

        v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pin.integrate(model, q, v * DT)
        i += 1


def solve_ik_trajectory(model, data, joint_id, poses, q_init, print_every=10, **ik_kwargs):
    """Solve IK for a sequence of task-space poses, warm starting each solve with
    the previous solution so that consecutive configurations stay in the same
    redundancy branch instead of jumping between valid solutions. That is what
    keeps the resulting joint-space trajectory smooth.

    Prints the iteration count and error norm every print_every waypoints, or not
    at all with print_every=0.

    Returns (q_traj, successes), a list of q per pose and a list of bools per
    pose.
    """
    q = q_init.copy()
    q_traj = []
    successes = []
    for k, oMdes in enumerate(poses):
        q, success, err, iters = solve_ik(model, data, joint_id, oMdes, q, **ik_kwargs)
        q_traj.append(q.copy())
        successes.append(success)
        if print_every and k % print_every == 0:
            print(f"waypoint {k:4d}: iters={iters:4d}  |err|={norm(err):.2e}  success={success}")
    return q_traj, successes
