import numpy as np
from scipy.linalg import expm
import pinocchio as pin


def linearize_continuous(model, data, q, v, tau):
    """Continuous-time linearization of the rigid-body dynamics about
    (q, v, tau): xdot = f(x, u), x = (q, v), u = tau.

    xdot = [v, a(q, v, tau)], so:
        d(xdot)/dq   = [0,      ddq_dq]
        d(xdot)/dv   = [I,      ddq_dv]
        d(xdot)/dtau = [0,      ddq_dtau]

    Returns (A_c, B_c): A_c is (2*nv, 2*nv), B_c is (2*nv, nv).
    """
    nv = model.nv
    ddq_dq, ddq_dv, ddq_dtau = pin.computeABADerivatives(model, data, q, v, tau)

    A_c = np.block([
        [np.zeros((nv, nv)), np.eye(nv)],
        [ddq_dq, ddq_dv],
    ])
    B_c = np.vstack([np.zeros((nv, nv)), ddq_dtau])

    return A_c, B_c


def discretize(A_c, B_c, dt):
    """Zero-order-hold discretization of (A_c, B_c) using the augmented matrix
    exponential. This is exact if the linearized dynamics are constant over the
    step, rather than the cruder first-order A_d = I + A_c*dt approximation.

    Returns (A_d, B_d).
    """
    n = A_c.shape[0]
    m = B_c.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_c * dt
    M[:n, n:] = B_c * dt
    E = expm(M)
    A_d = E[:n, :n]
    B_d = E[:n, n:]
    return A_d, B_d


def linearize_discrete(model, data, q, v, tau, dt):
    """Continuous linearization about (q, v, tau), then zero-order-hold
    discretization at timestep dt. Returns (A_d, B_d).
    """
    A_c, B_c = linearize_continuous(model, data, q, v, tau)
    return discretize(A_c, B_c, dt)


def linearize_trajectory(model, data, q_traj, v_traj, tau_traj, dt):
    """linearize_discrete at every point along a trajectory.

    q_traj, v_traj, tau_traj: sequences of length N (arrays or lists of
    per-timestep vectors).

    Returns (A_traj, B_traj): lists of length N, one A_d (2*nv, 2*nv) /
    B_d (2*nv, nv) per timestep.
    """
    A_traj = []
    B_traj = []
    for q, v, tau in zip(q_traj, v_traj, tau_traj):
        A_d, B_d = linearize_discrete(model, data, q, v, tau, dt)
        A_traj.append(A_d)
        B_traj.append(B_d)
    return A_traj, B_traj
